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

    def clean(self, text):
        return str(text).strip().replace('\n', ' ').replace('\r', '').strip()

    def extract_case_info(self):
        df = self.df

        # Fixed fields by position
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

        # Expand with synonyms
        expanded_key_map = {}
        for field, syns in key_map.items():
            all_syns = set(s.lower() for s in syns)
            for s in syns:
                all_syns.update(get_wordnet_synonyms(s))
            expanded_key_map[field] = list(all_syns)

        all_synonyms = []
        synonym_to_field = {}
        for field, syns in expanded_key_map.items():
            for syn in syns:
                all_synonyms.append(syn)
                synonym_to_field[syn] = field

        for i in range(len(df)):
            row_raw = [str(x).strip().lower() for x in df.iloc[i, :12] if pd.notna(x)]
            if any("qty" in cell and "item" in cell for cell in row_raw):
                break  # Stop when materials table begins

            row = [self.clean(str(x)) for x in df.iloc[i, :12] if pd.notna(x)]
            for idx, cell in enumerate(row):
                cell_lower = cell.lower()

                # ✅ Extract all known technical fields from cell (regardless of match)
                multi_matches = {
                    "cautery": re.search(r"cautery:\s*([^\s]+)", cell, re.I),
                    "bipolar": re.search(r"bipolar:\s*([^\s]+)", cell, re.I),
                    "sponge count": re.search(r"sponge count:\s*(\w+)", cell, re.I),
                    "ligasure": re.search(r"ligasure:\s*([^\s]+)", cell, re.I)
                }
                for key, match in multi_matches.items():
                    if match:
                        case_info[key] = match.group(1)

                # ✅ Continue fuzzy keyword match for general fields
                best_match, score, _ = process.extractOne(cell_lower, all_synonyms, scorer=fuzz.partial_ratio)
                if score >= 85:
                    field = synonym_to_field[best_match]

                    # Skip if it's a technical field already processed above
                    if field in multi_matches:
                        continue

                    if ':' in cell:
                        value = cell.split(":", 1)[1].strip()
                    elif idx + 1 < len(row):
                        value = row[idx + 1].strip()
                    else:
                        value = ""

                    if not case_info[field]:
                        case_info[field] = value

        return case_info

    def extract_other_keys(self):
        wanted_keys = ["Instruments Pulled By", "Supplies Pulled By", "LAST UPDATED"]
        other_keys = {}
        for _, row in self.df.iterrows():
            for val in row:
                sval = self.clean(val)
                if ":" in sval:
                    key, value = [v.strip() for v in sval.split(":", 1)]
                    if key in wanted_keys:
                        if re.fullmatch(r"_+", value):
                            value = ""
                        other_keys[key] = value
        return other_keys

    def extract_material_tables(self):
        processed_headers = set()
        rows, cols = self.df.shape

        for row in range(rows):
            col = 0
            while col < cols:
                header_key = (row, col)
                if header_key in processed_headers:
                    col += 1
                    continue

                if str(self.df.iloc[row, col]).strip() == "QTY":
                    for search_col in range(col + 1, min(col + 10, cols)):
                        if str(self.df.iloc[row, search_col]).strip() == "USED":
                            start_col = col
                            end_col = search_col + 1

                            processed_headers.add(header_key)
                            start_row = row + 1
                            end_row = start_row

                            while end_row < rows:
                                cell = str(self.df.iloc[end_row, start_col]).strip()
                                last_cell = str(self.df.iloc[end_row, end_col - 1]).strip()
                                if cell == "QTY" and last_cell == "USED":
                                    break
                                end_row += 1

                            data_block = self.df.iloc[start_row:end_row, start_col:end_col].copy()
                            header = self.df.iloc[row, start_col:end_col].tolist()
                            data_block.columns = header
                            data_block = data_block.loc[:, ~pd.isna(data_block.columns)]
                            data_block.columns = [str(col).strip() for col in data_block.columns]

                            exclude_keywords = [
                                "last updated:",
                                "instruments pulled by:",
                                "supplies pulled by:",
                                "patient label"
                            ]

                            def row_contains_keywords(row):
                                for cell in row:
                                    if isinstance(cell, str):
                                        for keyword in exclude_keywords:
                                            if keyword in cell.lower():
                                                return True
                                return False

                            data_block = data_block[~data_block.apply(row_contains_keywords, axis=1)]
                            data_block.dropna(how='all', inplace=True)
                            data_block.reset_index(drop=True, inplace=True)

                            self.tables.append(data_block)
                            col = end_col
                            break
                    else:
                        col += 1
                else:
                    col += 1

    def clean_notes_from_tables(self):
        notes_keywords = ["circulator notes", "surgical tech notes"]
        cleaned_tables = []

        for table in self.tables:
            note_start_idx = None
            note_header = None

            for i, row in table.iterrows():
                for cell in row:
                    if isinstance(cell, str):
                        for keyword in notes_keywords:
                            if keyword in cell.lower():
                                note_start_idx = i
                                note_header = keyword.title()
                                break
                if note_start_idx is not None:
                    break

            if note_start_idx is not None:
                self.notes_table = table.iloc[note_start_idx:].copy()
                self.notes_table.dropna(axis=1, how='all', inplace=True)

                if self.notes_table.shape[1] > 1:
                    self.notes_table = self.notes_table.apply(
                        lambda row: ' '.join(str(x) for x in row if pd.notna(x)), axis=1
                    ).to_frame()

                self.notes_table.columns = [note_header]
                self.notes_table = self.notes_table.iloc[1:].copy()

                top_table = table.iloc[:note_start_idx].copy()
                top_table.dropna(how='all', inplace=True)
                if not top_table.empty:
                    cleaned_tables.append(top_table)
            else:
                cleaned_tables.append(table)

        self.tables = cleaned_tables

    def convert_tables_to_json(self):
        for table in self.tables:
            if table.empty:
                continue

            columns = [col.lower() if isinstance(col, str) else str(col).lower() for col in table.columns]
            if len(columns) < 2:
                continue

            group_key = columns[-2].strip().lower()
            if group_key not in self.final_json:
                self.final_json[group_key] = []

            for _, row in table.iterrows():
                item = {}
                for col_name, value in zip(columns, row):
                    item[col_name] = "" if pd.isna(value) else str(value).strip()
                self.final_json[group_key].append(item)

        lines = []
        for _, row in self.notes_table.iterrows():
            row_text = " ".join(str(cell).strip() for cell in row if pd.notna(cell)).strip()
            if row_text:
                lines.append(row_text)
        notes_text = "\n".join(lines)
        self.final_json["Surgical Tech Notes"] = [{"notes": notes_text}]

    def extract_all(self):
        case_info = self.extract_case_info(self.df) 
        other_keys = self.extract_other_keys()
        self.extract_material_tables()
        self.clean_notes_from_tables()
        self.convert_tables_to_json()

        output = {
            "Case Info": case_info,
            "Other Keys": other_keys,
        }
        output.update(self.final_json)

        with open("output_caseinfo_and_notes.json", "w") as f:
            json.dump(output, f, indent=2)

        print("✅ Extraction complete. Output saved to output_caseinfo_and_notes.json")



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
            # Read all sheet names
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
                # Add sheet status for debugging
                if (not case_info or all(v == '' for v in case_info.values())) and not extractor.final_json:
                    sheet_status = "empty or unrecognized format"
                else:
                    sheet_status = "data extracted"
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
