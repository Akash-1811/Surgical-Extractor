import pandas as pd
import json
import re
from rapidfuzz import process, fuzz
from nltk.corpus import wordnet as wn
import nltk
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import tempfile

# Download WordNet if not already present
nltk.download('wordnet')

class SurgicalCaseExtractor:
    def __init__(self, file_path, sheet_name=None):
        self.file_path = file_path
        if sheet_name is not None:
            self.df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        else:
            self.df = pd.read_excel(file_path, header=None)
        self.tables = []
        self.notes_table = pd.DataFrame()
        self.final_json = {}

    def clean(self, val):
        if pd.isna(val):
            return ""
        return str(val).strip()

    def extract_case_info(self, df):
        case_info = {
            "surgeon": self.clean(df.iloc[2, 0]),
            "procedure": self.clean(df.iloc[3, 0]),
            "position": "",
            "prep": "",
            "assistant": "",
            "gloves": "",
            "sponge count": "",
            "cautery": "",
            "bipolar": "",
            "ligasure": ""
        }

        key_map = {
            "position": ["position", "patient position", "pos", "setup"],
            "prep": ["prep", "preparation", "skin prep", "pre-op prep"],
            "assistant": ["assistant", "assist", "asst"],
            "gloves": ["gloves", "glove size", "glove", "hand size"],
            "sponge count": ["sponge count", "sponges", "sponge", "count"],
            "cautery": ["cautery", "bovie", "electrocautery"],
            "bipolar": ["bipolar"],
            "ligasure": ["ligasure", "liga sure", "ligasure device"]
        }

        def get_wordnet_synonyms(word):
            synonyms = set()
            for syn in wn.synsets(word.replace(" ", "_")):
                for lemma in syn.lemmas():
                    synonyms.add(lemma.name().replace("_", " ").lower())
            return synonyms

        expanded_key_map = {}
        for field, syns in key_map.items():
            all_syns = set(syns)
            for s in syns:
                all_syns.update(get_wordnet_synonyms(s))
            expanded_key_map[field] = list(all_syns)

        all_synonyms = []
        synonym_to_field = {}
        for field, syns in expanded_key_map.items():
            for syn in syns:
                all_synonyms.append(syn)
                synonym_to_field[syn] = field

        for i in range(10):
            row = [self.clean(x) for x in df.iloc[i, :12]]
            for idx, cell in enumerate(row):
                cell_lower = cell.lower()

                multi_matches = {
                    "cautery": re.search(r"cautery[:\s]*([^\s]+)", cell, re.I),
                    "bipolar": re.search(r"bipolar[:\s]*([^\s]+)", cell, re.I),
                    "sponge count": re.search(r"sponge count[:\s]*([^\s]+)", cell, re.I),
                }
                for field, match in multi_matches.items():
                    if match and not case_info[field]:
                        case_info[field] = match.group(1)

                best_match, score, _ = process.extractOne(cell_lower, all_synonyms, scorer=fuzz.partial_ratio)
                if score >= 85:
                    outk = synonym_to_field[best_match]
                    if outk in multi_matches:
                        continue

                    if ":" in cell:
                        value = cell.split(":", 1)[1].strip()
                    elif idx + 1 < len(row):
                        value = row[idx + 1].strip()
                    else:
                        value = ""

                    if not case_info[outk]:
                        case_info[outk] = value

        return case_info

    # [Remaining methods unchanged for brevity: extract_other_keys, extract_material_tables, clean_notes_from_tables, convert_tables_to_json, extract_all]

api_v1 = FastAPI()

@api_v1.post("/surgical-extract")
def extract_excel(file: UploadFile = File(...)):
    if not file.filename.endswith((".xlsx", ".xls", ".csv")):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an Excel or CSV file.")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename[file.filename.rfind('.'):]) as tmp:
            tmp.write(file.file.read())
            tmp_path = tmp.name
        if file.filename.endswith((".xlsx", ".xls")):
            xls = pd.ExcelFile(tmp_path)
            result = {}
            for sheet_name in xls.sheet_names:
                extractor = SurgicalCaseExtractor(tmp_path, sheet_name=sheet_name)
                case_info = extractor.extract_case_info(extractor.df)
                other_keys = extractor.extract_other_keys()
                extractor.extract_material_tables()
                extractor.clean_notes_from_tables()
                extractor.convert_tables_to_json()
                output = {
                    "Case Info": case_info,
                    "Other Keys": other_keys,
                }
                output.update(extractor.final_json)
                sheet_status = "data extracted" if case_info or extractor.final_json else "empty or unrecognized format"
                result[sheet_name] = {
                    "sheet_status": sheet_status,
                    "extracted": output
                }
            return JSONResponse(content=result)
        elif file.filename.endswith(".csv"):
            class CSVSurgicalCaseExtractor(SurgicalCaseExtractor):
                def __init__(self, file_path):
                    self.file_path = file_path
                    self.df = pd.read_csv(file_path, header=None)
                    self.tables = []
                    self.notes_table = pd.DataFrame()
                    self.final_json = {}
            extractor = CSVSurgicalCaseExtractor(tmp_path)
            case_info = extractor.extract_case_info(extractor.df)
            other_keys = extractor.extract_other_keys()
            extractor.extract_material_tables()
            extractor.clean_notes_from_tables()
            extractor.convert_tables_to_json()
            output = {
                "Case Info": case_info,
                "Other Keys": other_keys,
            }
            output.update(extractor.final_json)
            return JSONResponse(content=output)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("uploader:api_v1", host="0.0.0.0", port=8000)
