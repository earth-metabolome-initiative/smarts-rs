//! Live PostgreSQL storage for SMARTS queries.

use diesel::{Connection, PgConnection as TestConnection};

include!("diesel_common/text_storage.rs");

#[test]
#[ignore = "requires a live PostgreSQL server and POSTGRES_DATABASE_URL"]
fn postgres_stores_and_loads_smarts_text() {
    let database_url = std::env::var("POSTGRES_DATABASE_URL").expect("set POSTGRES_DATABASE_URL");
    let mut connection = TestConnection::establish(&database_url).expect("PostgreSQL connection");
    check_text_storage(&mut connection);
}
