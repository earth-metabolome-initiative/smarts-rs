//! In-memory SQLite storage for SMARTS queries.

use diesel::{Connection, SqliteConnection as TestConnection};

include!("diesel_common/text_storage.rs");

#[test]
fn sqlite_stores_and_loads_smarts_text() {
    let mut connection = TestConnection::establish(":memory:").expect("SQLite connection");
    check_text_storage(&mut connection);
}
