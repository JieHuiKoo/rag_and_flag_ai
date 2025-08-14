# %% This file reads content from wikipedia and downloads
# the content of all country pages.

# %% Import wikipedia API
import wikipedia
# %% Import other libraries

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
def get_country_content(country):
    """
    Fetches the content of a Wikipedia page for a given country.
    """
    try:
        page = wikipedia.page(country)
        return page.content
    except wikipedia.exceptions.PageError:
        print(f"Page not found for {country}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# %%

list_of_countries = get_list_of_countries("list_of_countries.txt")

# %%

for country in list_of_countries:
    content = get_country_content(country)
    if content:
        print(f"Content for {country}:\n{content[:500]}...\n")  # Print first 500 characters
    else:
        print(f"No content found for {country}\n")

    