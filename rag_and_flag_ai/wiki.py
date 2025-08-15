# %% This file reads content from wikipedia and downloads
# the content of all country pages.

# %% Import wikipedia API
import wikipedia
# %% Import other libraries
import pandas as pd

# %% Read the list of countries from a text file
def get_list_of_countries(file_path):
    """
    Reads a text file containing a list of countries.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        countries = file.read().splitlines()
        countries = countries[0]
        list_of_countries = countries.split(",")
        countries = []
        # Split each country by the pipe character and strip whitespace
        for country in list_of_countries:
            country = country.split("|")
            if len(country) > 1:
                countries.append(country[1].strip())

    return countries

# %% Fetch content for each country from Wikipedia
def get_country_content(query):
    """
    Fetches the content of a Wikipedia page for a given country.
    """
    max_results = 10  # Maximum number of results to return
    keywords = ["country"]

    wikipedia.set_lang("en")  # Set the language to English
    

    try:
        results = wikipedia.search(query, results=max_results)
        print(results)
        filtered = [r for r in results if any(k.lower() in r.lower() for k in keywords)]
        best_match = filtered[0] if filtered else results[0]
        print(f"✅ Best match: {best_match}")

        # Fetch the page
        page = wikipedia.page(best_match ,auto_suggest=False)

        return page.content
    except wikipedia.exceptions.PageError:
        print(f"Page not found for {query}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# %%

def generate_country_content():
    list_of_countries = get_list_of_countries("list_of_countries.txt")
        
    # Fetch content for each country
    country_content = {}
    for country in list_of_countries:
        print(f"Fetching content for {country}...")
        content = get_country_content(country)
        country_content[country] = content

    # The following countries have special cases
    country_content["Libyan Arab Jamahiriya"] = get_country_content("Libya")
    country_content["Macedonia"] = get_country_content("Macedonia (region)")
    country_content["Congo"] = get_country_content("Democratic Republic of the Congo")
    # Generate the dataframe
    country_content_df = pd.DataFrame(country_content.items(), columns=["Country", "Content"])

    return country_content_df

# %%
content_df = generate_country_content()
# %%
# Save the content to a CSV file
content_df.to_csv("country_content.csv", index=False)
print("Content saved to country_content.csv")