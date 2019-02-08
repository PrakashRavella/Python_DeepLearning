import urllib.request
from bs4 import BeautifulSoup
import os

wikiURL = "https://en.wikipedia.org/wiki/Machine_learning"
openURL = urllib.request.urlopen(wikiURL)

# Assigning Parsed Web Page into a Variable
soup = BeautifulSoup(openURL, "html.parser")

# Title of the Page
title = soup.find('title')
print("Title --> ", title.text)

# Finding Links in WebPage
links = soup.find_all('a')

# Iterating the Links and showing Href values of the anchor tag
print("Links -->")
for link in links:
    print(link.get('href'))
