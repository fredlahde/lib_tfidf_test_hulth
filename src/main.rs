#[macro_use]
extern crate serde_derive;
extern crate lib_tfidf;
extern crate serde;
extern crate serde_json;

use lib_tfidf::{Document, Tfidf, Token};

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fs;
use std::io::{self, Read};
use std::path::Path;

pub type HulthDocumentKeywords = HashMap<String, Vec<Vec<String>>>;

#[derive(Debug, Serialize, Deserialize)]
pub struct HulthDocument {
    #[serde(rename = "sentences")]
    pub(crate) sentences: Vec<Sentence>,
}

impl HulthDocument {
    fn get_flat_tokens(&self) -> Vec<Box<HulthToken>> {
        let mut ret = vec![];
        for s in &self.sentences {
            s.tokens
                .iter()
                .map(|t| (*t).clone())
                .map(|t| Box::new(t))
                .for_each(|t| ret.push(t));
        }
        ret
    }
}

impl Document<String, HulthToken> for HulthDocument {
    fn get_id(&self) -> Box<String> {
        Box::new("".into())
    }

    fn get_content(&self) -> Vec<Box<HulthToken>> {
        self.get_flat_tokens()
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Sentence {
    #[serde(rename = "tokens")]
    pub(crate) tokens: Vec<HulthToken>,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename = "token")]
pub struct HulthToken {
    #[serde(rename = "word")]
    pub(crate) word: String,

    #[serde(rename = "lemma")]
    pub(crate) lemma: String,

    #[serde(rename = "offsetBegin")]
    pub(crate) offset_begin: i64,

    #[serde(rename = "offsetEnd")]
    pub(crate) offset_end: i64,

    #[serde(rename = "pos")]
    pub(crate) pos: String,
}

impl Token for HulthToken {
    fn get_term(&self) -> String {
        self.word.clone()
    }
    fn get_offset_begin(&self) -> usize {
        self.offset_begin as usize
    }
    fn get_pos(&self) -> Option<String> {
        None
    }
}

impl HulthToken {
    // TODO return a view instead of allocating new memory
    fn clone(&self) -> Self {
        HulthToken {
            word: self.word.clone(),
            lemma: self.lemma.clone(),
            offset_begin: self.offset_begin,
            offset_end: self.offset_end,
            pos: self.pos.clone(),
        }
    }
}

/// iterates over all files in directory non-recursively
/// and applies f
/// returns an Err on first Err returned from f
fn for_each_file<P, F>(path: P, mut f: F) -> io::Result<()>
where
    P: AsRef<Path>,
    F: FnMut(&Path) -> io::Result<()>,
{
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            continue;
        } else {
            f(&path)?
        }
    }
    Ok(())
}

struct MeasureHolder {
    precision: f64,
    recall: f64,
    f1: f64,
}

fn main() -> io::Result<()> {
    let mut docs: Vec<Box<dyn Document<String, HulthToken>>> = vec![];
    let dir = "dataset/testJSON";
    for_each_file(dir, |path| {
        let mut json = String::new();
        std::fs::File::open(path)?.read_to_string(&mut json)?;
        let doc: HulthDocument = serde_json::from_str(&json)?;
        docs.push(Box::new(doc));
        Ok(())
    })?;

    let mut tfidf = Tfidf::new(docs.as_slice());
    tfidf.fit_transform()?;

    let keywords: HulthDocumentKeywords =
        serde_json::from_reader(std::fs::File::open("dataset/references/test.uncontr.json")?)?;

    let mut measures = vec![];
    for_each_file(dir, |path| {
        let mut json = String::new();
        std::fs::File::open(path)?.read_to_string(&mut json)?;
        let doc: HulthDocument = serde_json::from_str(&json)?;
        let ranked = tfidf.rank_tokens(&doc.get_content())?;
        let mut ranked = ranked.iter().collect::<Vec<(_, _)>>();
        ranked.sort_by(|a, b| cmp_f64(*a.1, *b.1));

        let name = path
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or_default();
        let name = name.replace(".json", "");
        let reference = keywords.get(&name);
        if let Some(reference) = reference {
            let reference = reference
                .iter()
                .flat_map(|v| v.iter().flat_map(|s| s.split(" ")))
                .collect::<Vec<_>>();
            let mut relevant = vec![];
            for (term, _) in ranked.iter() {
                if reference.contains(&term.as_str()) {
                    relevant.push(term);
                }
            }
            let precision = relevant.len() as f64 / ranked.len() as f64;
            let recall = relevant.len() as f64 / reference.len() as f64;
            measures.push(MeasureHolder {
                precision: precision,
                recall: recall,
                f1: f1(precision, recall),
            });
        } else {
            eprintln!("{}", name);
            return Err(io::Error::new(io::ErrorKind::Other, "found no keywords"));
        }
        Ok(())
    })?;

    let precision_mean = mean(&measures.iter().map(|m| m.precision).collect::<Vec<f64>>());
    let recall_mean = mean(&measures.iter().map(|m| m.recall).collect::<Vec<f64>>());
    let f1_mean = mean(&measures.iter().map(|m| m.f1).collect::<Vec<f64>>());
    println!(
        "precision: {} recall {} f1 {}",
        precision_mean, recall_mean, f1_mean
    );

    Ok(())
}

fn mean(v: &[f64]) -> f64 {
    let sum: f64 = v.iter().sum();
    sum / v.len() as f64
}

fn f1(precision: f64, recall: f64) -> f64 {
    if precision == 0f64 || recall == 0f64 {
        return 0f64;
    }
    let tmp = (precision * recall) / (precision + recall);
    2f64 * tmp
}

#[allow(clippy::comparison_chain)]
fn cmp_f64(a: f64, b: f64) -> Ordering {
    if a.is_nan() {
        return Ordering::Less;
    }
    if b.is_nan() {
        return Ordering::Greater;
    }
    if a < b {
        return Ordering::Greater;
    } else if a > b {
        return Ordering::Less;
    }
    Ordering::Equal
}
