//! Extract the `text` column from a HuggingFace parquet file into a plain text file.
//!
//! Usage:
//!   cargo run --release --example extract_parquet_text -- input.parquet output.txt

use parquet::file::reader::{FileReader, SerializedFileReader};
use std::fs::File;
use std::io::{BufWriter, Write};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let input = args.get(1).expect("usage: extract_parquet_text <in.parquet> <out.txt>");
    let output = args.get(2).expect("usage: extract_parquet_text <in.parquet> <out.txt>");

    let file = File::open(input)?;
    let reader = SerializedFileReader::new(file)?;

    let out_file = File::create(output)?;
    let mut writer = BufWriter::new(out_file);

    let mut row_count = 0usize;
    let mut byte_count = 0usize;

    for row_result in reader.get_row_iter(None)? {
        let row = row_result?;
        for (name, field) in row.get_column_iter() {
            if name == "text" {
                let s = match field {
                    parquet::record::Field::Str(s) => s.as_str(),
                    _ => continue,
                };
                if s.is_empty() { continue; }
                writer.write_all(s.as_bytes())?;
                writer.write_all(b"\n")?;
                byte_count += s.len() + 1;
                row_count += 1;
            }
        }
    }

    writer.flush()?;
    println!("Wrote {} rows, {} bytes ({:.2} MB) to {}",
        row_count, byte_count, byte_count as f64 / 1_048_576.0, output);
    Ok(())
}
