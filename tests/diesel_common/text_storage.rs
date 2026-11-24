use diesel::{connection::SimpleConnection, prelude::*, result::Error::DeserializationError};
use smarts_rs::{QueryMol, SmartsParseError, SmartsParseErrorKind};

mod schema {
    diesel::table! {
        smarts_queries (id) {
            id -> Integer,
            pattern -> Text,
        }
    }
}

fn check_text_storage(connection: &mut TestConnection) {
    use schema::smarts_queries;

    connection
        .batch_execute(
            "CREATE TEMPORARY TABLE smarts_queries (id INTEGER PRIMARY KEY, pattern TEXT NOT NULL)",
        )
        .expect("create temporary SMARTS table");

    for (identifier, source, expected) in [(1, "C=C", "C=C"), (2, "[$([OH])]C", "[$([O&H])]C")] {
        let query = source.parse::<QueryMol>().expect("parse SMARTS");
        diesel::insert_into(smarts_queries::table)
            .values((
                smarts_queries::id.eq(identifier),
                smarts_queries::pattern.eq(&query),
            ))
            .execute(connection)
            .expect("insert query");
        let row = smarts_queries::table
            .find(identifier)
            .select(smarts_queries::pattern);
        assert_eq!(
            row.first::<String>(connection).expect("read stored text"),
            expected
        );
        assert_eq!(
            row.first::<QueryMol>(connection).expect("read typed query"),
            query
        );
    }

    diesel::insert_into(smarts_queries::table)
        .values((smarts_queries::id.eq(3), smarts_queries::pattern.eq("[O")))
        .execute(connection)
        .expect("insert malformed SMARTS");
    let error = smarts_queries::table
        .find(3)
        .select(smarts_queries::pattern)
        .first::<QueryMol>(connection)
        .expect_err("reject malformed SMARTS");
    let DeserializationError(cause) = error else {
        panic!("expected a deserialization error, got {error}");
    };
    let cause: &(dyn std::error::Error + 'static) = cause.as_ref();
    let parse_error = std::iter::successors(Some(cause), |&cause| cause.source())
        .find_map(|cause| cause.downcast_ref::<SmartsParseError>())
        .expect("retain the SMARTS parse error");
    assert_eq!(
        parse_error.kind(),
        SmartsParseErrorKind::UnterminatedBracketAtom
    );
}
