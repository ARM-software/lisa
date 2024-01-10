use std::{fs::File, path::Path};

use arrow2::{
    array::{Array, TryPush},
    chunk::Chunk,
    datatypes::{DataType, Field, Schema},
    error::Result,
    io::ipc::write,
};
use arrow2_convert::{
    field::ArrowField,
    serialize::{ArrowSerialize, TryIntoArrow},
};
use futures::{stream::Stream, StreamExt as FuturesStreamExt};

use crate::string::String;

macro_rules! newtype_impl_arrow_field {
    ($type:ty, $inner:ty) => {
        impl ::arrow2_convert::field::ArrowField for $type {
            type Type = Self;
            fn data_type() -> arrow2::datatypes::DataType {
                <$inner as ::arrow2_convert::field::ArrowField>::data_type()
            }
        }

        impl ::arrow2_convert::serialize::ArrowSerialize for $type {
            type MutableArrayType =
                <$inner as ::arrow2_convert::serialize::ArrowSerialize>::MutableArrayType;

            #[inline]
            fn new_array() -> Self::MutableArrayType {
                Self::MutableArrayType::default()
            }

            #[inline]
            fn arrow_serialize(
                v: &Self,
                array: &mut Self::MutableArrayType,
            ) -> arrow2::error::Result<()> {
                use arrow2::array::TryPush;
                array.try_push(Some(v.0.clone()))
            }
        }

        impl ::arrow2_convert::deserialize::ArrowDeserialize for $type {
            type ArrayType = <$inner as ::arrow2_convert::deserialize::ArrowDeserialize>::ArrayType;

            #[inline]
            fn arrow_deserialize(
                v: <&Self::ArrayType as ::std::iter::IntoIterator>::Item,
            ) -> ::std::option::Option<<Self as ::arrow2_convert::field::ArrowField>::Type> {
                v.map(|t| t.clone().into())
            }
        }
    };
}
pub(crate) use newtype_impl_arrow_field;

impl arrow2_convert::field::ArrowField for String {
    type Type = Self;
    fn data_type() -> arrow2::datatypes::DataType {
        arrow2::datatypes::DataType::Utf8
    }
}

impl arrow2_convert::serialize::ArrowSerialize for String {
    type MutableArrayType = arrow2::array::MutableUtf8Array<i32>;

    #[inline]
    fn new_array() -> Self::MutableArrayType {
        Self::MutableArrayType::default()
    }

    #[inline]
    fn arrow_serialize(v: &Self, array: &mut Self::MutableArrayType) -> arrow2::error::Result<()> {
        array.try_push(Some(v))
    }
}

impl arrow2_convert::deserialize::ArrowDeserialize for String {
    type ArrayType = arrow2::array::Utf8Array<i32>;

    #[inline]
    fn arrow_deserialize(v: Option<&str>) -> Option<Self> {
        v.map(|t| t.into())
    }
}

fn get_arrow_type<Element>() -> DataType
where
    Element: ArrowSerialize + ArrowField<Type = Element> + 'static,
{
    let empty: [Element; 0] = [];
    let array: Box<dyn Array> = empty.try_into_arrow().unwrap();
    array.data_type().clone()
}

fn get_schema<Element>() -> Schema
where
    Element: ArrowSerialize + ArrowField<Type = Element> + 'static,
{
    let typ = get_arrow_type::<Element>();
    let field = Field::new("", typ, false);
    Schema::from(vec![field])
}
pub async fn write_stream<P, S>(file_path: P, stream: S) -> Result<()>
where
    P: AsRef<Path>,
    S: Stream,
    S::Item: ArrowSerialize + ArrowField<Type = S::Item> + 'static,
{
    const CHUNK_SIZE: usize = 128 * 1024;

    // Using BufWriter on top did not seem to improve much, probably because we
    // write in chunks anyway.
    let file = File::create(file_path)?;

    let schema = get_schema::<S::Item>();
    // LZ4 is fast enough to not be noticible (or at least fast enough to make
    // up for the lost time in I/O speed).
    // However, it is not yet available for WASM:
    // https://github.com/jorgecarleitao/arrow2/issues/986
    let compression = Some(write::Compression::LZ4);
    // let compression = None;
    let options = write::WriteOptions { compression };
    let mut writer = write::FileWriter::new(file, schema, None, options);

    writer.start()?;

    let process = |vec: Result<Vec<S::Item>>, item: S::Item| {
        let vec = match vec {
            Ok(mut vec) => {
                vec.push(item);

                // TODO: We could delegate the writing to another thread and
                // just send the vector over a channel
                if vec.len() > CHUNK_SIZE {
                    let array: Box<dyn Array> = vec.try_into_arrow().unwrap();
                    let chunk = Chunk::new(vec![array]);
                    match writer.write(&chunk, None) {
                        Ok(_) => Ok(Vec::with_capacity(CHUNK_SIZE)),
                        Err(err) => Err(err),
                    }
                } else {
                    Ok(vec)
                }
            }
            vec => vec,
        };

        async move { vec }
    };

    let vec = Vec::with_capacity(CHUNK_SIZE);
    let vec = stream.fold(Ok(vec), process).await?;

    // Write the last bits
    let array: Box<dyn Array> = vec.try_into_arrow().unwrap();
    let chunk = Chunk::new(vec![array]);
    writer.write(&chunk, None)?;

    writer.finish()?;

    Ok(())
}
