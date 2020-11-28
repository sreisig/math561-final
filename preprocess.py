import pandas as pd

def extract_eco1_windspeed_fuelmoisture():
    """
    Extracts average windspeed and fuel moisture by level 1 ecoregion (data originally segmented by level 3 ecoregion)
    Assumes that fm_ws_monthly_ecn.csv has been moved from Nagy's repo into data/
    """
    df = pd.read_csv('./data/fm_ws_monthly_ecn.csv')
    df['NA_L1CODE'] = df['NA_L3CODE'].str.split('.').map(lambda x:x[0])
    l1_ecoregion_vals = df.groupby('NA_L1CODE').mean()
    return l1_ecoregion_vals
    
def save_eco1_windspeed_fuelmoisture():
    """
    Saves avg windspeed, fuelmoisture to file
    """
    df = extract_eco1_windspeed_fuelmoisture()
    df.to_csv('./data/l1_windspeed_fuelmoisture.csv')

if __name__ == "__main__":
    save_eco1_windspeed_fuelmoisture()
