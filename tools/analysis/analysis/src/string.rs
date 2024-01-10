use core::{
    fmt,
    fmt::Display,
    ops::{Deref, DerefMut},
};

use schemars::{gen::SchemaGenerator, schema::Schema, JsonSchema};
use serde::{Deserialize, Serialize};

// type StringImplem = std::string::String;
type StringImplem = smartstring::alias::String;

#[derive(Clone, Debug, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct String(StringImplem);

impl String {
    #[inline]
    pub fn new() -> Self {
        String(StringImplem::new())
    }
}

impl From<&str> for String {
    #[inline]
    fn from(s: &str) -> String {
        String(s.into())
    }
}

impl PartialEq<String> for String {
    #[inline]
    fn eq(&self, other: &String) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T> PartialEq<T> for String
where
    StringImplem: PartialEq<T>,
{
    #[inline]
    fn eq(&self, other: &T) -> bool {
        self.0.eq(&other)
    }
}

impl Deref for String {
    type Target = StringImplem;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for String {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl AsRef<str> for String {
    #[inline]
    fn as_ref(&self) -> &str {
        self
    }
}

impl JsonSchema for String {
    #[inline]
    fn schema_name() -> std::string::String {
        <std::string::String as JsonSchema>::schema_name()
    }
    #[inline]
    fn json_schema(gen: &mut SchemaGenerator) -> Schema {
        <std::string::String as JsonSchema>::json_schema(gen)
    }
    #[inline]
    fn is_referenceable() -> bool {
        <std::string::String as JsonSchema>::is_referenceable()
    }
}
impl Display for String {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}
