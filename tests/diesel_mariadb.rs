//! Live MariaDB storage for SMARTS queries.

use diesel::{Connection, MariadbConnection as TestConnection};

include!("diesel_common/text_storage.rs");

#[test]
#[ignore = "requires a live MariaDB server and MARIADB_DATABASE_URL"]
fn mariadb_stores_and_loads_smarts_text() {
    let database_url = std::env::var("MARIADB_DATABASE_URL").expect("set MARIADB_DATABASE_URL");
    let mut connection = TestConnection::establish(&database_url).expect("MariaDB connection");
    check_text_storage(&mut connection);
}
