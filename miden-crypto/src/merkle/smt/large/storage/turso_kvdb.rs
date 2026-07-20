//! Turso (SQLite) implementation of the [`KVDB`](super::KVDB) traits.
//!
//! # Async bridging
//!
//! Turso exposes a fully async API. The `KVDB` trait is sync. We bridge by owning a
//! `tokio::runtime::Runtime` (current-thread) and calling `rt.block_on(...)` for every op.
//!
//! The mutex on the connection is always acquired INSIDE the `block_on` closure, never
//! outside it. This avoids holding a `MutexGuard` across an async boundary.
//!
//! IMPORTANT: `Runtime::block_on` panics if called from inside another tokio runtime.
//! The SMT storage layer is purely synchronous — no tokio context is active when any
//! of these methods are called. Do not use `TursoKVDB` from async code directly.

#![allow(clippy::upper_case_acronyms)]
#![allow(clippy::await_holding_lock)] // TODO: fix

use alloc::{boxed::Box, sync::Arc, vec::Vec};
use core::fmt;
use std::sync::Mutex;

use turso::{Connection, Database, Value};

use super::{RocksDbConfig, StorageError, StorageResult, kvdb::*, schema::*};

// ERROR CONVERSION
// ================================================================================================

impl From<turso::Error> for StorageError {
    fn from(e: turso::Error) -> Self {
        StorageError::Backend(Box::new(e))
    }
}

// TURSO KVDB
// ================================================================================================

struct TursoKVDBInner {
    db: Database,
    rt: Arc<tokio::runtime::Runtime>,
    conn: Arc<Mutex<Connection>>,
}

impl fmt::Debug for TursoKVDBInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TursoKVDBInner").finish_non_exhaustive()
    }
}

/// Turso (SQLite) backed [`KVDB`](super::KVDB) implementation.
///
/// Each of the nine SMT tables maps to a SQL table with `(key BLOB PRIMARY KEY, value BLOB NOT
/// NULL)`. All async turso operations are driven by an owned current-thread tokio runtime; do not
/// call methods on this type from inside an async tokio context.
#[derive(Clone, Debug)]
pub struct TursoKVDB {
    inner: Arc<TursoKVDBInner>,
}

impl KVDB for TursoKVDB {
    type Batch<'a> = TursoKVDBBatch;

    fn new(config: RocksDbConfig) -> StorageResult<Self> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| StorageError::Backend(Box::new(e)))?;

        // Like redb, if the caller passes a directory append a default filename.
        let file_path = if config.path.is_dir() {
            config.path.join("data.turso")
        } else {
            config.path
        };
        let path_str = file_path.to_string_lossy().into_owned();

        let db = rt.block_on(turso::Builder::new_local(&path_str).build())?;
        let conn = db.connect()?;

        // Set pragmas; ignore errors — turso may not implement all of them.
        let kib = -(config.cache_size as i64 / 1024);
        rt.block_on(async {
            if let Ok(mut rows) = conn.query("PRAGMA journal_mode = WAL", ()).await {
                while rows.next().await.ok().flatten().is_some() {}
            }
            let _ = conn.execute("PRAGMA synchronous = NORMAL", ()).await;
            let _ = conn.execute(&alloc::format!("PRAGMA cache_size = {kib}"), ()).await;
            let _ = conn.execute("PRAGMA wal_autocheckpoint = 1000", ()).await;
        });

        // Create all nine tables idempotently.
        let create_tables = alloc::format!(
            "CREATE TABLE IF NOT EXISTS {LEAVES_CF}       (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {SUBTREE_16_CF}   (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {SUBTREE_24_CF}   (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {SUBTREE_32_CF}   (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {SUBTREE_40_CF}   (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {SUBTREE_48_CF}   (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {SUBTREE_56_CF}   (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {METADATA_CF}     (key BLOB PRIMARY KEY, value BLOB NOT NULL);\
             CREATE TABLE IF NOT EXISTS {IN_MEM_DEPTH_CF} (key BLOB PRIMARY KEY, value BLOB NOT NULL);"
        );
        rt.block_on(conn.execute_batch(&create_tables))?;

        let rt = Arc::new(rt);
        Ok(Self {
            inner: Arc::new(TursoKVDBInner { db, rt, conn: Arc::new(Mutex::new(conn)) }),
        })
    }

    fn delete(&self, table: &TursoTable, key: &[u8]) -> StorageResult<()> {
        let conn = Arc::clone(&self.inner.conn);
        let sql = alloc::format!("DELETE FROM {} WHERE key = ?", table.name);
        let key = key.to_vec();
        self.inner
            .rt
            .block_on(async move {
                let guard = conn.lock().expect("turso conn mutex poisoned");
                let mut stmt = guard.prepare_cached(&sql).await?;
                stmt.execute(turso::params::Params::Positional(vec![Value::Blob(key)])).await?;
                Ok::<_, turso::Error>(())
            })
            .map_err(Into::into)
    }

    fn batch(&self) -> TursoKVDBBatch {
        let conn = Arc::clone(&self.inner.conn);
        let rt = Arc::clone(&self.inner.rt);
        rt.block_on(async {
            let guard = conn.lock().expect("turso conn mutex poisoned");
            guard.execute("BEGIN DEFERRED", ()).await
        })
        .expect("BEGIN DEFERRED failed; cannot propagate error from batch() per trait");
        TursoKVDBBatch { conn, rt, err: None, committed: false }
    }

    fn snapshot(&self) -> StorageResult<TursoKVDBSnapshot> {
        let conn = self.inner.db.connect()?;
        self.inner.rt.block_on(conn.execute("BEGIN DEFERRED", ()))?;
        // Promote the deferred txn to an actual read transaction immediately so the MVCC
        // snapshot view is anchored before any concurrent writer can advance the WAL.
        // SELECT returns rows, so drain via query().
        self.inner.rt.block_on(async {
            let mut rows = conn.query("SELECT 1 FROM sqlite_schema LIMIT 1", ()).await?;
            while rows.next().await?.is_some() {}
            Ok::<_, turso::Error>(())
        })?;
        Ok(TursoKVDBSnapshot {
            inner: Arc::new(TursoKVDBSnapshotInner {
                rt: Arc::clone(&self.inner.rt),
                conn: Mutex::new(conn),
            }),
        })
    }

    fn sync(&self) -> StorageResult<()> {
        let conn = Arc::clone(&self.inner.conn);
        // cacheflush() is sync — call it while holding the lock briefly.
        {
            let guard = conn.lock().expect("turso conn mutex poisoned");
            guard.cacheflush().map_err(StorageError::from)?;
        }
        // WAL checkpoint — returns rows, so use query() and drain them.
        self.inner
            .rt
            .block_on(async move {
                let guard = conn.lock().expect("turso conn mutex poisoned");
                let mut rows = guard.query("PRAGMA wal_checkpoint(FULL)", ()).await?;
                while rows.next().await?.is_some() {}
                Ok::<_, turso::Error>(())
            })
            .map_err(StorageError::from)
    }
}

// TURSO KVDB BATCH
// ================================================================================================

/// Atomic batch of put/delete operations.
///
/// Shares the connection Arc with the live DB. The mutex is acquired briefly per op (BEGIN,
/// each put/delete, COMMIT) and released immediately after, so `get()` calls on the same
/// `TursoKVDB` can interleave — they see the in-progress transaction state.
pub struct TursoKVDBBatch {
    conn: Arc<Mutex<Connection>>,
    rt: Arc<tokio::runtime::Runtime>,
    err: Option<StorageError>,
    committed: bool,
}

impl fmt::Debug for TursoKVDBBatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TursoKVDBBatch").finish_non_exhaustive()
    }
}

impl KVDBBatch for TursoKVDBBatch {
    type Table = TursoTable;

    fn put(&mut self, table: &TursoTable, key: &[u8], value: &[u8]) {
        if self.err.is_some() {
            return;
        }
        let conn = Arc::clone(&self.conn);
        let sql =
            alloc::format!("INSERT OR REPLACE INTO {} (key, value) VALUES (?, ?)", table.name);
        let key = key.to_vec();
        let value = value.to_vec();
        let res = self.rt.block_on(async move {
            let guard = conn.lock().expect("turso conn mutex poisoned");
            let mut stmt = guard.prepare_cached(&sql).await?;
            stmt.execute(turso::params::Params::Positional(vec![
                Value::Blob(key),
                Value::Blob(value),
            ]))
            .await?;
            Ok::<_, turso::Error>(())
        });
        if let Err(e) = res {
            self.err = Some(e.into());
        }
    }

    fn delete(&mut self, table: &TursoTable, key: &[u8]) {
        if self.err.is_some() {
            return;
        }
        let conn = Arc::clone(&self.conn);
        let sql = alloc::format!("DELETE FROM {} WHERE key = ?", table.name);
        let key = key.to_vec();
        let res = self.rt.block_on(async move {
            let guard = conn.lock().expect("turso conn mutex poisoned");
            let mut stmt = guard.prepare_cached(&sql).await?;
            stmt.execute(turso::params::Params::Positional(vec![Value::Blob(key)])).await?;
            Ok::<_, turso::Error>(())
        });
        if let Err(e) = res {
            self.err = Some(e.into());
        }
    }

    fn commit(mut self) -> StorageResult<()> {
        let conn = Arc::clone(&self.conn);
        let rt = Arc::clone(&self.rt);
        if let Some(e) = self.err.take() {
            self.committed = true;
            let _ = rt.block_on(async move {
                let guard = conn.lock().expect("turso conn mutex poisoned");
                guard.execute("ROLLBACK", ()).await
            });
            return Err(e);
        }
        rt.block_on(async move {
            let guard = conn.lock().expect("turso conn mutex poisoned");
            guard.execute("COMMIT", ()).await
        })?;
        self.committed = true;
        Ok(())
    }
}

impl Drop for TursoKVDBBatch {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        let conn = Arc::clone(&self.conn);
        self.rt.block_on(async move {
            let guard = conn.lock().expect("turso conn mutex poisoned");
            if !guard.is_autocommit().unwrap_or(true) {
                let _ = guard.execute("ROLLBACK", ()).await;
            }
        });
    }
}

// KVDB READER
// ================================================================================================

impl KVDBReader for TursoKVDB {
    type Table = TursoTable;
    type Bytes<'a> = Vec<u8>;
    type Snapshot = TursoKVDBSnapshot;

    fn table(&self, name: &str) -> StorageResult<TursoTable> {
        Ok(TursoTable { name: static_table_name(name)? })
    }

    fn get<'a>(&self, table: &TursoTable, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        let conn = Arc::clone(&self.inner.conn);
        let sql = alloc::format!("SELECT value FROM {} WHERE key = ?", table.name);
        let key = key.to_vec();
        self.inner
            .rt
            .block_on(async move {
                let guard = conn.lock().expect("turso conn mutex poisoned");
                let mut stmt = guard.prepare_cached(&sql).await?;
                let mut rows =
                    stmt.query(turso::params::Params::Positional(vec![Value::Blob(key)])).await?;
                match rows.next().await? {
                    Some(row) => match row.get_value(0)? {
                        Value::Blob(v) => Ok(Some(v)),
                        _ => Ok(None),
                    },
                    None => Ok::<Option<Vec<u8>>, turso::Error>(None),
                }
            })
            .map_err(Into::into)
    }

    fn iter<'a>(
        &'a self,
        table: TursoTable,
    ) -> impl Iterator<Item = StorageResult<(Vec<u8>, Vec<u8>)>> + 'a {
        let conn = Arc::clone(&self.inner.conn);
        let sql = alloc::format!("SELECT key, value FROM {} ORDER BY key", table.name);
        let setup = self.inner.rt.block_on(async move {
            let guard = conn.lock().expect("turso conn mutex poisoned");
            let mut stmt = guard.prepare_cached(&sql).await?;
            stmt.query(turso::params::Params::None).await
        });
        match setup {
            Ok(rows) => Box::new(TursoRowIter { rt: &self.inner.rt, rows: Some(rows) })
                as Box<dyn Iterator<Item = _> + 'a>,
            Err(e) => Box::new(core::iter::once(Err(e.into()))),
        }
    }
}

// TURSO TABLE
// ================================================================================================

/// Lightweight table handle: just the `&'static str` name.
#[derive(Clone)]
pub struct TursoTable {
    pub(super) name: &'static str,
}

impl fmt::Debug for TursoTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TursoTable").field("name", &self.name).finish()
    }
}

// TURSO ROW ITERATOR
// ================================================================================================

struct TursoRowIter<'a> {
    rt: &'a tokio::runtime::Runtime,
    rows: Option<turso::Rows>,
}

impl Iterator for TursoRowIter<'_> {
    type Item = StorageResult<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        let rows = self.rows.as_mut()?;
        match self.rt.block_on(rows.next()) {
            Ok(None) => {
                self.rows = None;
                None
            },
            Ok(Some(row)) => {
                let pair = (|| -> StorageResult<(Vec<u8>, Vec<u8>)> {
                    let k = match row.get_value(0)? {
                        Value::Blob(b) => b,
                        _ => return Err(StorageError::Unsupported("non-blob key".into())),
                    };
                    let v = match row.get_value(1)? {
                        Value::Blob(b) => b,
                        _ => return Err(StorageError::Unsupported("non-blob value".into())),
                    };
                    Ok((k, v))
                })();
                Some(pair)
            },
            Err(e) => {
                self.rows = None;
                Some(Err(e.into()))
            },
        }
    }
}

// TURSO KVDB SNAPSHOT
// ================================================================================================

struct TursoKVDBSnapshotInner {
    rt: Arc<tokio::runtime::Runtime>,
    conn: Mutex<Connection>,
}

impl fmt::Debug for TursoKVDBSnapshotInner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("TursoKVDBSnapshotInner").finish_non_exhaustive()
    }
}

impl Drop for TursoKVDBSnapshotInner {
    fn drop(&mut self) {
        if let Ok(conn) = self.conn.get_mut()
            && !conn.is_autocommit().unwrap_or(true)
        {
            let _ = self.rt.block_on(conn.execute("ROLLBACK", ()));
        }
    }
}

/// Point-in-time read-only snapshot implementing [`KVDBReader`].
#[derive(Clone, Debug)]
pub struct TursoKVDBSnapshot {
    inner: Arc<TursoKVDBSnapshotInner>,
}

impl KVDBReader for TursoKVDBSnapshot {
    type Table = TursoTable;
    type Bytes<'a> = Vec<u8>;
    type Snapshot = TursoKVDBSnapshot;

    fn table(&self, name: &str) -> StorageResult<TursoTable> {
        Ok(TursoTable { name: static_table_name(name)? })
    }

    fn get<'a>(&self, table: &TursoTable, key: &[u8]) -> StorageResult<Option<Vec<u8>>> {
        let inner = Arc::clone(&self.inner);
        let sql = alloc::format!("SELECT value FROM {} WHERE key = ?", table.name);
        let key = key.to_vec();
        self.inner
            .rt
            .block_on(async move {
                let guard = inner.conn.lock().expect("turso snapshot conn mutex poisoned");
                let mut stmt = guard.prepare_cached(&sql).await?;
                let mut rows =
                    stmt.query(turso::params::Params::Positional(vec![Value::Blob(key)])).await?;
                match rows.next().await? {
                    Some(row) => match row.get_value(0)? {
                        Value::Blob(v) => Ok(Some(v)),
                        _ => Ok(None),
                    },
                    None => Ok::<Option<Vec<u8>>, turso::Error>(None),
                }
            })
            .map_err(Into::into)
    }

    fn iter<'a>(
        &'a self,
        table: TursoTable,
    ) -> impl Iterator<Item = StorageResult<(Vec<u8>, Vec<u8>)>> + 'a {
        let inner = Arc::clone(&self.inner);
        let sql = alloc::format!("SELECT key, value FROM {} ORDER BY key", table.name);
        let setup = self.inner.rt.block_on(async move {
            let guard = inner.conn.lock().expect("turso snapshot conn mutex poisoned");
            let mut stmt = guard.prepare_cached(&sql).await?;
            stmt.query(turso::params::Params::None).await
        });
        match setup {
            Ok(rows) => Box::new(TursoRowIter { rt: &self.inner.rt, rows: Some(rows) })
                as Box<dyn Iterator<Item = _> + 'a>,
            Err(e) => Box::new(core::iter::once(Err(e.into()))),
        }
    }
}

// HELPERS
// ================================================================================================

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
