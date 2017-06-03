import requests
import json
from bs4 import BeautifulSoup

def scrapeLink(link):
    if "nytimes.com" in link:
        return getFromNewYorkTimes(link)
    elif "washingtonpost.com" in link:
        return getFromWashingtonPost(link)
    elif "bbc.com" in link:
        return getFromBBC(link)
    elif "bostonglobe.com" in link:
        return getFromBOGLO(link)
    elif "politico.com" in link:
        return getFromPOLITICO(link)
    elif "cnn.com" in link:
        return getFromCNN(link)
    else:
        raise Exception("No scraper for link" + link)

def useLinkFile(path):
    links = []
    with open(path) as f:
        for line in f.readlines():
            links.append(" ".join(line.split()))

    articles = []
    for link in links:
        try:
            article = scrapeLink(link)
        except:
            print(link)
            raise RuntimeError
        if article != {}:
            articles.append(article)

    return articles

def getFromCNN(link):
    html = requests.get(link).text
    soup = BeautifulSoup(html, "lxml")

    content = {}

    article = soup.find("div", {"class": "zn-body__read-all"})
    try:
        headline = soup.find("h1", {"class" : "pg-headline"}).text
    except:
        print(link)
        return

    storybody = soup.find_all(attrs={"class": "zn-body__paragraph"})
    wholestory = ""
    for p in storybody:
        wholestory = wholestory + " " + p.text

    content["headline"] = headline.split()
    content["body"] = wholestory.split()
    content["link"] = link
    return content
    

def getFromPOLITICO(link):
    html = requests.get(link).text
    soup = BeautifulSoup(html, "lxml")

    content = {}

    article = soup.find("div", {"class": "story-text"})
    try:
        headline = article.find("h1", {"class" : " "}).text
    except:
        print("DID NOT WORK: " + link)
        return

    storybody = article.find_all("p", {"class": None})
    wholestory = ""
    for p in storybody:
        wholestory = wholestory + " " + p.text

    content["headline"] = headline.split()
    content["body"] = wholestory.split()
    content["link"] = link
    return content


def getFromBBC(link):
    html = requests.get(link).text
    soup = BeautifulSoup(html, "lxml")

    content = {}

    article = soup.find("div", {"class": "story-body"})

    headline = article.find("h1").text

    storybody = article.find("div", {"class": "story-body__inner"}).find_all("p", {"class": None})
    wholestory = ""
    for p in storybody:
        wholestory = wholestory + " " + p.text

    content["headline"] = headline.split()
    content["body"] = wholestory.split()
    content["link"] = link
    return content

def getFromBOGLO(link):
    html = requests.get(link).text
    soup = BeautifulSoup(html, "lxml")

    content = {}

    article = soup.find("div", {"class" : "article-text"})
    storybody = article.find_all("p", {"class" : None})
    headline = soup.find("h1", {"class" : "main-hed"})

    wholestory = ""
    for p in storybody:
        wholestory = wholestory + " " + p.text

    content["headline"] = headline.text.split()
    content["body"] = wholestory.split()
    content["link"] = link
    return content

def getFromNewYorkTimes(link):
    html = requests.get(link).text
    soup = BeautifulSoup(html, "lxml")

    content = {}

    headline = soup.find(id="headline").text

    article = soup.article
    storybody = article.find_all("p", {"class": "story-body-text story-content"})
    wholestory = ""
    for p in storybody:
        wholestory = wholestory + " " + p.text

    content["headline"] = headline.split()
    content["body"] = wholestory.split()
    content["link"] = link

    return content

def getFromWashingtonPost(link):
    html = requests.get(link).text
    soup = BeautifulSoup(html, "lxml")

    content = {}

    headlinewrapper = soup.find(id="top-content")
    headline = headlinewrapper.find_all("h1")[0].text

    article = soup.article
    storybody = article.find_all("p", {"class": None})
    wholestory = ""
    for p in storybody:
        wholestory = wholestory + " " + p.text

    content["headline"] = headline.split()
    content["body"] = wholestory.split()
    content["link"] = link

    return content

def saveLinkList(path):
    newPath = path + ".saved"
    articles = useLinkFile(path)
    with open(newPath, "a") as f:
        for article in articles:
                f.write(json.dumps(article))
                f.write("\n")
        f.close()
