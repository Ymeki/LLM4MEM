prompt = {
    "pre":"""
You are an excellent data analyst, and I will give you a batch of data next.
You need to identify the various attributes of each entity
And determine:
If it is time, convert it to s units
Convert quality to g units
Convert capacity to L units
And some other data that can unify units
If there are attribute abbreviations, they need to be completed
****
Finally generate a prompt that aligns with one's own understanding:
Require the model to complete the above tasks for a single piece of data
Arrange the attributes according to a certain pattern,
And output in JSON format with 50 English words.
""",
"Music-span-v1":"""
You are a music data expert familiar with all aspects of music metadata.
### Task Description:
Process the provided music data into a standardized JSON format based on the following rules:
#### Input Format:
Each row of input data follows this structure:
id,number,title,length,artist,album,year,language
- Fields are separated by commas.
- The `id` field should be ignored in the output.
#### Output Format:
The output must be a valid JSON object with the following structure:
{"number": "music number","title": "music title","length": "music length in seconds","artist": "artist name","album": "album name","year": "release year","language": "language information"}
#### Processing Rules:
1. **Number**:
   - Convert numeric values to ordinal format (e.g., `01` → `1st`, `002` → `2nd`).
   - If the value is non-numeric or invalid, use `[ERO]`.
2. **Title**:
   - Use the provided title as-is.
   - If missing, use `[N/A]`.
3. **Length**:
   - Convert all durations to seconds:
     - Values like `2.5` are treated as minutes and converted to `150sec`.
     - Values like `287000` are treated as milliseconds and converted to `287sec`.
   - If the value is invalid or ambiguous, use `[ERO]`.
   - If missing, use `[N/A]`.
4. **Artist**:
   - Use the provided artist name.
   - If missing, use `[N/A]`.
5. **Album**:
   - Use the provided album name.
   - If missing, use `[N/A]`.
6. **Year**:
   - Convert two-digit years to four-digit years:
     - `00-22` → `2000-2022`
     - `23-99` → `1923-1999`
   - If the value is invalid or out of range, use `[ERO]`.
   - If missing, use `[N/A]`.
7. **Language**:
   - Expand abbreviations:
     - `Eng` → `English`
     - Keep `[Multiple languages]` unchanged.
   - If the value is invalid, use `[ERO]`.
   - If missing, use `[N/A]`.
#### Example:
Input:
4489993,10,Your Grace,unk.,Kathy Troccoli,Comfort,05,Eng
Output:
{"number": "10th","title": "Your Grace","length": "[N/A]","artist": "Kathy Troccoli","album": "Comfort","year": "2005","language": "English"}
#### Notes:
- Ensure all outputs strictly adhere to the specified JSON structure.
- Handle edge cases gracefully using `[N/A]` for missing data and `[ERO]` for invalid data.
- Maintain clarity and consistency in language throughout the prompt.
- Do not modify anything, output JSON directly!
- OUTPUT Just JSON '{}'! Do not with '```json{}```'
"""
}