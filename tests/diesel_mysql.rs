//! Live MySQL storage for SMARTS queries.

use diesel::{Connection, MysqlConnection as TestConnection};

include!("diesel_common/text_storage.rs");

#[test]
#[ignore = "requires a live MySQL server and MYSQL_DATABASE_URL"]
fn mysql_stores_and_loads_smarts_text() {
    let database_url = std::env::var("MYSQL_DATABASE_URL").expect("set MYSQL_DATABASE_URL");
    let mut connection = TestConnection::establish(&database_url).expect("MySQL connection");
    check_text_storage(&mut connection);
}
