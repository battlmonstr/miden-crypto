//! Redb implementation of the [`KVDB`](super::KVDB) traits.

#![allow(clippy::upper_case_acronyms)]

use alloc::{boxed::Box, sync::Arc};
use core::{fmt, ops::Deref};

use redb::{
    AccessGuard, Database, Durability, ReadTransaction, ReadableDatabase, TableDefinition,
    WriteTransaction,
};

use super::{RocksDbConfig, StorageError, StorageResult, kvdb::*, schema::*};

// ERROR CONVERSIONS
// ================================================================================================

macro_rules! redb_into_storage_error {
    ($($t:ty),*) => {
        $(impl From<$t> for StorageError {
            fn from(e: $t) -> Self { StorageError::Backend(Box::new(e)) }
        })*
    };
}

redb_into_storage_error!(
    redb::DatabaseError,
    redb::TransactionError,
    redb::TableError,
    redb::StorageError,
    redb::CommitError
);

// REDB KVDB
// ================================================================================================

struct RedbKVDBInner {
    db: Database,
}

impl fmt::Debug for RedbKVDBInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedbKVDBInner").finish_non_exhaustive()
    }
}

/// Redb-backed [`KVDB`](super::KVDB) implementation.
///
/// Wraps a `Database` inside an `Arc<RedbKVDBInner>`. Cloning is cheap (just bumps the Arc).
/// All nine SMT tables are pre-created on `new()` so reader paths never hit `TableDoesNotExist`.
#[derive(Clone, Debug)]
pub struct RedbKVDB {
    inner: Arc<RedbKVDBInner>,
}

impl KVDB for RedbKVDB {
    type Batch<'a> = RedbKVDBBatch;

    /// Opens or creates a redb database at the path given in `config`.
    ///
    /// `max_open_files` is intentionally ignored — redb is a single-file DB with no
    /// equivalent concept. `cache_size` is wired to `Database::builder().set_cache_size()`.
    fn new(config: RocksDbConfig) -> StorageResult<Self> {
        let mut builder = Database::builder();
        builder.set_cache_size(config.cache_size);
        // RocksDB and fjall use a directory as the DB root; redb is a single file.
        // If the caller passes a directory, append a default filename so behavior is compatible.
        let file_path = if config.path.is_dir() {
            config.path.join("data.redb")
        } else {
            config.path
        };
        let db = builder.create(file_path)?;

        const ALL_TABLE_NAMES: &[&str] = &[
            LEAVES_CF,
            SUBTREE_16_CF,
            SUBTREE_24_CF,
            SUBTREE_32_CF,
            SUBTREE_40_CF,
            SUBTREE_48_CF,
            SUBTREE_56_CF,
            METADATA_CF,
            IN_MEM_DEPTH_CF,
        ];

        // Pre-create all nine tables so subsequent reader paths can open_table unconditionally.
        // redb returns TableDoesNotExist if a table has never been written.
        {
            let txn = db.begin_write()?;
            for name in ALL_TABLE_NAMES {
                let def = TableDefinition::<&[u8], &[u8]>::new(name);
                txn.open_table(def)?;
            }
            txn.commit()?;
        }

        Ok(Self { inner: Arc::new(RedbKVDBInner { db }) })
    }

    fn delete(&self, table: &RedbTable, key: &[u8]) -> StorageResult<()> {
        let txn = self.inner.db.begin_write()?;
        {
            let def = TableDefinition::<&[u8], &[u8]>::new(table.name);
            let mut t = txn.open_table(def)?;
            t.remove(key)?;
        }
        txn.commit()?;
        Ok(())
    }

    fn batch(&self) -> RedbKVDBBatch {
        let mut txn =
            self.inner.db.begin_write().expect(
                "redb begin_write failed; this should not happen with serialized SMT writes",
            );
        // Hot-path commits skip fsync; durability is established by sync().
        txn.set_durability(Durability::None)
            .expect("set_durability before any write cannot fail");
        RedbKVDBBatch { txn: Some(txn), err: None }
    }

    fn snapshot(&self) -> StorageResult<RedbKVDBSnapshot> {
        let txn = self.inner.db.begin_read()?;
        Ok(RedbKVDBSnapshot { txn: Arc::new(txn) })
    }

    fn sync(&self) -> StorageResult<()> {
        // Force fsync of any prior Durability::None batch commits via an empty
        // Durability::Immediate commit (Immediate is the default on a new txn).
        let txn = self.inner.db.begin_write()?;
        txn.commit()?;
        Ok(())
    }
}

// REDB KVDB BATCH
// ================================================================================================

/// Atomic batch of put/delete operations. Wraps a live `WriteTransaction` with
/// `Durability::None`; call [`RedbKVDBBatch::commit`] to flush and (later) `sync()` to fsync.
pub struct RedbKVDBBatch {
    // Option so we can take ownership in commit(self); WriteTransaction::commit consumes by value.
    txn: Option<WriteTransaction>,
    // Sticky error from a failed put/delete (trait methods return (), so we stash here).
    err: Option<StorageError>,
}

impl fmt::Debug for RedbKVDBBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedbKVDBBatch").finish_non_exhaustive()
    }
}

impl KVDBBatch for RedbKVDBBatch {
    type Table = RedbTable;

    fn put(&mut self, table: &RedbTable, key: &[u8], value: &[u8]) {
        if self.err.is_some() {
            return;
        }
        let Some(txn) = self.txn.as_ref() else { return };
        let def = TableDefinition::<&[u8], &[u8]>::new(table.name);
        let res = (|| -> StorageResult<()> {
            let mut t = txn.open_table(def)?;
            t.insert(key, value)?;
            Ok(())
        })();
        if let Err(e) = res {
            self.err = Some(e);
        }
    }

    fn delete(&mut self, table: &RedbTable, key: &[u8]) {
        if self.err.is_some() {
            return;
        }
        let Some(txn) = self.txn.as_ref() else { return };
        let def = TableDefinition::<&[u8], &[u8]>::new(table.name);
        let res = (|| -> StorageResult<()> {
            let mut t = txn.open_table(def)?;
            t.remove(key)?;
            Ok(())
        })();
        if let Err(e) = res {
            self.err = Some(e);
        }
    }

    fn commit(mut self) -> StorageResult<()> {
        if let Some(e) = self.err.take() {
            // Drop self.txn → redb aborts the uncommitted write transaction.
            return Err(e);
        }
        let txn = self.txn.take().expect("batch already committed");
        txn.commit()?;
        Ok(())
    }
}

// KVDB READER
// ================================================================================================

impl KVDBReader for RedbKVDB {
    type Table = RedbTable;
    type Bytes<'a> = RedbBytes;
    type Snapshot = RedbKVDBSnapshot;

    fn table(&self, name: &str) -> StorageResult<RedbTable> {
        Ok(RedbTable { name: static_table_name(name)? })
    }

    fn get<'a>(&self, table: &RedbTable, key: &[u8]) -> StorageResult<Option<RedbBytes>> {
        let txn = self.inner.db.begin_read()?;
        let def = TableDefinition::<&[u8], &[u8]>::new(table.name);
        let t = txn.open_table(def)?;
        Ok(t.get(key)?.map(RedbBytes))
    }

    fn iter<'a>(
        &'a self,
        table: RedbTable,
    ) -> impl Iterator<Item = StorageResult<(RedbBytes, RedbBytes)>> + 'a {
        iter_table(&self.inner.db, table.name)
    }
}

// REDB TABLE
// ================================================================================================

/// Lightweight table handle: just the `&'static str` name.
///
/// Unlike fjall (which caches `Arc<KeyspaceInner>`) or RocksDB (which holds a CF pointer),
/// redb table handles only exist inside a transaction — so we remember only the name and
/// reconstruct the handle per operation.
#[derive(Clone)]
pub struct RedbTable {
    pub(super) name: &'static str,
}

impl fmt::Debug for RedbTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedbTable").field("name", &self.name).finish()
    }
}

// REDB BYTES
// ================================================================================================

/// Owned byte container returned by `RedbKVDB` reads.
///
/// Wraps `AccessGuard<'static, &'static [u8]>` — the `'static` lifetime parameters indicate
/// that the guard carries its own `Arc<TransactionGuard>` and keeps the `ReadTransaction` alive
/// independently. `Deref` exposes the underlying `&[u8]` via `.value()`.
pub struct RedbBytes(AccessGuard<'static, &'static [u8]>);

impl fmt::Debug for RedbBytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("RedbBytes").field(&self.deref()).finish()
    }
}

impl Deref for RedbBytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.0.value()
    }
}

impl AsRef<[u8]> for RedbBytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.deref()
    }
}

// REDB KVDB SNAPSHOT
// ================================================================================================

/// Point-in-time read-only snapshot implementing [`KVDBReader`].
///
/// Wraps an owned `ReadTransaction` (no borrowed lifetime — it holds `Arc<TransactionalMemory>`
/// internally), so no `ManuallyDrop`/`transmute` is needed unlike `rocksdb::Snapshot<'a>`.
#[derive(Clone)]
pub struct RedbKVDBSnapshot {
    txn: Arc<ReadTransaction>,
}

impl fmt::Debug for RedbKVDBSnapshot {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RedbKVDBSnapshot").finish_non_exhaustive()
    }
}

impl KVDBReader for RedbKVDBSnapshot {
    type Table = RedbTable;
    type Bytes<'a> = RedbBytes;
    type Snapshot = RedbKVDBSnapshot;

    fn table(&self, name: &str) -> StorageResult<RedbTable> {
        Ok(RedbTable { name: static_table_name(name)? })
    }

    fn get<'a>(&self, table: &RedbTable, key: &[u8]) -> StorageResult<Option<RedbBytes>> {
        let def = TableDefinition::<&[u8], &[u8]>::new(table.name);
        let t = self.txn.open_table(def)?;
        Ok(t.get(key)?.map(RedbBytes))
    }

    fn iter<'a>(
        &'a self,
        table: RedbTable,
    ) -> impl Iterator<Item = StorageResult<(RedbBytes, RedbBytes)>> + 'a {
        let setup = (|| -> StorageResult<_> {
            let def = TableDefinition::<&[u8], &[u8]>::new(table.name);
            let t = self.txn.open_table(def)?;
            // Use the inherent `ReadOnlyTable::range` (returns Range<'static, ...> via
            // internal Arc<TransactionGuard>) rather than the trait `iter()` which returns
            // Range<'_, ...> borrowing `t`.
            Ok(t.range::<&[u8]>(..)?)
        })();
        match setup {
            Ok(range) => Box::new(
                range.map(|res| res.map_err(Into::into).map(|(k, v)| (RedbBytes(k), RedbBytes(v)))),
            ) as Box<dyn Iterator<Item = _>>,
            Err(e) => Box::new(core::iter::once(Err(e))),
        }
    }
}

// HELPERS
// ================================================================================================

/// Validates `name` against the schema and returns the corresponding `&'static str`.
///
/// `KVDBReader::table` receives a `&str` (any lifetime) but `RedbTable` must store
/// `&'static str` so it can be used to construct `TableDefinition` at any point.
/// This function matches against the known schema constants and returns the static
/// version — no allocation needed.
fn static_table_name(name: &str) -> StorageResult<&'static str> {
    match name {
        LEAVES_CF => Ok(LEAVES_CF),
        SUBTREE_16_CF => Ok(SUBTREE_16_CF),
        SUBTREE_24_CF => Ok(SUBTREE_24_CF),
        SUBTREE_32_CF => Ok(SUBTREE_32_CF),
        SUBTREE_40_CF => Ok(SUBTREE_40_CF),
        SUBTREE_48_CF => Ok(SUBTREE_48_CF),
        SUBTREE_56_CF => Ok(SUBTREE_56_CF),
        METADATA_CF => Ok(METADATA_CF),
        IN_MEM_DEPTH_CF => Ok(IN_MEM_DEPTH_CF),
        _ => Err(StorageError::Unsupported(alloc::format!("unknown table `{name}`"))),
    }
}

/// Opens a fresh `ReadTransaction`, iterates the named table, and returns a boxed iterator.
/// The `ReadOnlyTable::iter()` result is `Range<'static, ...>` — it carries its own
/// `Arc<TransactionGuard>`, so the txn and table can be dropped after extraction.
fn iter_table<'db>(
    db: &'db Database,
    name: &'static str,
) -> Box<dyn Iterator<Item = StorageResult<(RedbBytes, RedbBytes)>> + 'db> {
    let setup = (|| -> StorageResult<_> {
        let txn = db.begin_read()?;
        let def = TableDefinition::<&[u8], &[u8]>::new(name);
        let t = txn.open_table(def)?;
        // Use the inherent ReadOnlyTable::range (Range<'static, ...>) not the trait iter()
        // (Range<'_, ...>) so the result outlives the local `t` and `txn`.
        Ok(t.range::<&[u8]>(..)?)
    })();
    match setup {
        Ok(range) => Box::new(
            range.map(|res| res.map_err(Into::into).map(|(k, v)| (RedbBytes(k), RedbBytes(v)))),
        ),
        Err(e) => Box::new(core::iter::once(Err(e))),
    }
}
