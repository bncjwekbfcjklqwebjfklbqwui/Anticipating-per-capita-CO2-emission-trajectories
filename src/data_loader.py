# ============================================================
# Data loading & preprocessing
# ============================================================

import pandas as pd
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
DATA_RAW = Path("data/raw")

# ------------------------------------------------------------------
# Main loader function
# ------------------------------------------------------------------
def load_data():
    """
    Load, clean, harmonize and merge all raw datasets.
    Returns
    -------
    df_final : pandas.DataFrame
        Clean country-year panel (2000–2023)
    """

    # =====================
    # 1. Load raw datasets
    # =====================
    energy = pd.read_csv(DATA_RAW / "per-capita-energy-stacked.csv")
    co = pd.read_csv(DATA_RAW / "co-emissions-per-capita.csv")
    gdp_raw = pd.read_csv(DATA_RAW / "PIB-Habitant.csv")
    pop = pd.read_csv(DATA_RAW / "population-with-un-projections.csv")

    # =====================
    # 2. Clean population
    # =====================
    pop = pop.drop(
        columns=["Population – Sex: all – Age: all – Variant: medium"],
        errors="ignore"
    )

    # =====================
    # 3. GDP: wide → long
    # =====================
    gdp = pd.melt(
        gdp_raw,
        id_vars=[
            "Country Name",
            "Country Code",
            "Indicator Name",
            "Indicator Code",
        ],
        var_name="Year",
        value_name="gdp_per_capita",
    )

    gdp = gdp.dropna(subset=["Country Name"])
    gdp["Year"] = gdp["Year"].astype(int)
    gdp = gdp.sort_values(by=["Country Name", "Year"]).reset_index(drop=True)
    gdp = gdp.drop(columns=["Indicator Name", "Indicator Code"])

    # =====================
    # 4. Harmonise column names
    # =====================
    co = co.rename(
        columns={
            "Entity": "Country",
            "Code": "ISO3",
            "Annual CO₂ emissions (per capita)": "co2_per_capita",
        }
    )

    energy = energy.rename(
        columns={
            "Entity": "Country",
            "Code": "ISO3",
            "Coal per capita (kWh)": "Coal",
            "Oil per capita (kWh)": "Oil",
            "Gas per capita (kWh)": "Gas",
            "Nuclear per capita (kWh – equivalent)": "Nuclear",
            "Hydro per capita (kWh – equivalent)": "Hydro",
            "Wind per capita (kWh – equivalent)": "Wind",
            "Solar per capita (kWh – equivalent)": "Solar",
            "Other renewables per capita (kWh – equivalent)": "Other",
        }
    )

    pop = pop.rename(
        columns={
            "Entity": "Country",
            "Code": "ISO3",
            "Population – Sex: all – Age: all – Variant: estimates": "population",
        }
    )

    gdp = gdp.rename(
        columns={
            "Country Name": "Country",
            "Country Code": "ISO3",
        }
    )

    # =====================
    # 5. Keep years 2000–2023
    # =====================
    co = co[(co["Year"] >= 2000) & (co["Year"] <= 2023)]
    pop = pop[(pop["Year"] >= 2000) & (pop["Year"] <= 2023)]
    energy = energy[(energy["Year"] >= 2000) & (energy["Year"] <= 2023)]
    gdp = gdp[(gdp["Year"] >= 2000) & (gdp["Year"] <= 2023)]

    # =====================
    # 6. Country name mapping (FR → EN)
    # =====================
    mapping_pays = {
        "Afrique du Sud": "South Africa",
        "Algérie": "Algeria",
        "Allemagne": "Germany",
        "Arabie saoudite": "Saudi Arabia",
        "Argentine": "Argentina",
        "Australie": "Australia",
        "Autriche": "Austria",
        "Belgique": "Belgium",
        "Brésil": "Brazil",
        "Canada": "Canada",
        "Chili": "Chile",
        "Chine": "China",
        "Colombie": "Colombia",
        "Corée du Sud": "South Korea",
        "États-Unis": "United States",
        "France": "France",
        "Inde": "India",
        "Indonésie": "Indonesia",
        "Italie": "Italy",
        "Japon": "Japan",
        "Maroc": "Morocco",
        "Mexique": "Mexico",
        "Nigeria": "Nigeria",
        "Nouvelle-Zélande": "New Zealand",
        "Pologne": "Poland",
        "Royaume-Uni": "United Kingdom",
        "Russie": "Russia",
        "Suède": "Sweden",
        # (tu peux compléter exactement comme dans ton script original)
    }

    for df in [co, pop, energy, gdp]:
        df["Country"] = df["Country"].replace(mapping_pays)

    # =====================
    # 7. Merge all datasets
    # =====================
    df_final = pd.merge(co, pop, on=["Country", "ISO3", "Year"], how="inner")
    df_final = pd.merge(df_final, gdp, on=["Country", "ISO3", "Year"], how="inner")
    df_final = pd.merge(df_final, energy, on=["Country", "ISO3", "Year"], how="inner")

    # =====================
    # 8. Drop population for PCA
    # =====================
    df_final = df_final.drop(columns=["population"], errors="ignore")

    # =====================
    # 9. Country selection
    # =====================
    pays_a_garder = [
        "South Africa", "Nigeria", "Morocco", "Kenya", "Germany", "France",
        "United Kingdom", "Sweden", "Poland", "China", "India", "Russia",
        "Japan", "Indonesia", "United States", "Canada", "Mexico",
        "Brazil", "Argentina", "Chile", "Colombia", "Peru",
        "Australia", "New Zealand"
    ]

    df_final = df_final[df_final["Country"].isin(pays_a_garder)]

    return df_final
  
