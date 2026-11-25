use diesel::{
    backend::Backend,
    deserialize::{self, FromSql},
    serialize::{self, IsNull, Output, ToSql},
    sql_types::Text,
};

use crate::QueryMol;

#[cfg(feature = "diesel-sqlite")]
use diesel::sqlite::Sqlite;

#[cfg(feature = "diesel-sqlite")]
impl ToSql<Text, Sqlite> for QueryMol {
    fn to_sql<'bind>(&'bind self, output: &mut Output<'bind, '_, Sqlite>) -> serialize::Result {
        use alloc::string::ToString;

        output.set_value(self.to_string());
        Ok(IsNull::No)
    }
}

#[cfg(feature = "diesel-sqlite")]
impl FromSql<Text, Sqlite> for QueryMol {
    fn from_sql(mut value: <Sqlite as Backend>::RawValue<'_>) -> deserialize::Result<Self> {
        Ok(value.read_text().parse()?)
    }
}

#[cfg(any(
    feature = "diesel-postgres",
    feature = "diesel-mariadb",
    feature = "diesel-mysql"
))]
macro_rules! impl_server_text {
    ($backend:ty) => {
        impl ToSql<Text, $backend> for QueryMol {
            fn to_sql<'bind>(
                &'bind self,
                output: &mut Output<'bind, '_, $backend>,
            ) -> serialize::Result {
                use std::io::Write;

                write!(output, "{self}")?;
                Ok(IsNull::No)
            }
        }

        impl FromSql<Text, $backend> for QueryMol {
            fn from_sql(value: <$backend as Backend>::RawValue<'_>) -> deserialize::Result<Self> {
                Ok(core::str::from_utf8(value.as_bytes())?.parse()?)
            }
        }
    };
}

#[cfg(feature = "diesel-postgres")]
impl_server_text!(diesel::pg::Pg);
#[cfg(feature = "diesel-mariadb")]
impl_server_text!(diesel::mariadb::Mariadb);
#[cfg(feature = "diesel-mysql")]
impl_server_text!(diesel::mysql::Mysql);
