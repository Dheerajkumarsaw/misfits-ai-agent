import gspread
from oauth2client.service_account import ServiceAccountCredentials

def get_sheet_data(sheet_name, worksheet_index=0):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("google_creds.json", scope)
    client = gspread.authorize(creds)
    sheet = client.open(sheet_name).get_worksheet(worksheet_index)
    return sheet.get_all_records()  # Returns list of dicts
