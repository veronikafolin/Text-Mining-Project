from datasets.load import load_from_disk
import random
from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By

destinationPath = "../datasets/COMMA/"
split = "train"

dataset = load_from_disk(destinationPath + split)
print("\nLoaded dataset: ")
print(dataset)

# count unique values on dataset
uniqueValues = dataset.unique("codice_pronuncia")
print("\nUnique values: " + str(len(uniqueValues)))

# select 10 random examples
print("\n10 Random examples: ")
for _ in range(10):
    index = random.randint(0, len(dataset))
    print(dataset[index]["codice_pronuncia"])
    print(dataset[index]["judgement"])
    print(dataset[index]["constitutional_parameters"])

# examples with empty values on judgement attribute
emptyExamples = dataset.filter(lambda example: example["judgement"] == "")
print("\nExamples with no judgement: " + str(len(emptyExamples)))
print(emptyExamples["codice_pronuncia"])
for example in emptyExamples:
    print(example["codice_pronuncia"])

# examples with empty values on constitutional_parameters attribute
emptyExamples = dataset.filter(lambda example: example["constitutional_parameters"] == [[""]])
print("\nExamples with no constitutional parameters: " + str(len(emptyExamples)))
for _ in range(10):
    index = random.randint(0, len(emptyExamples))
    print(emptyExamples[index]["codice_pronuncia"])


def scrapeExample(codicePronuncia):
    judgement = ""
    constitutional_parameters = []

    # create Google web driver
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    wd = webdriver.Chrome(service=Service(r'chromedriver_win32/chromedriver.exe'), options=options)

    # open the web page
    wd.get("https://www.cortecostituzionale.it/actionPronuncia.do")

    # verify the correct loading of the web page
    def is_ready(wd):
        return wd.execute_script(r"""
                return document.readyState === 'complete'
            """)

    WebDriverWait(wd, 30).until(is_ready)

    idDecision, year = codicePronuncia.split("/")

    print("Scraping of " + codicePronuncia)

    try:
        # select the field of the year
        yearInput = wd.find_element(By.XPATH,
                                    "//*[@id='body']/section[2]/div/div/div/form/div/ul[2]/li[1]/div/div/div[1]/label/input")
        yearInput.send_keys(year)

        # select the field of the idDecision
        idDecisionInput = wd.find_element(By.XPATH,
                                          "//*[@id='body']/section[2]/div/div/div/form/div/ul[2]/li[1]/div/div/div[2]/label/input")
        idDecisionInput.send_keys(idDecision)

        # submit request
        submitButton = wd.find_element(By.XPATH,
                                       "//*[@id='body']/section[2]/div/div/div/form/div/ul[2]/li[1]/div/div/div[11]/button")
        wd.execute_script("arguments[0].click();", submitButton)

        # visualize the search result
        wd.find_element(By.XPATH, "/html/body/div[2]/div/form/section[2]/div/div/div/ul/li/span/a").click()
    except NoSuchElementException:
        print("ERROR on visualizing the search result  - Decision: " + idDecision + "/" + year)
        constitutional_parameters.append([""])
        return judgement, constitutional_parameters

    # get judgement
    try:
        judgement = wd.find_element(By.XPATH, "//*[@id='id1']/section[4]/div/div[1]/div[2]/strong").text
    except NoSuchElementException:
        print("ERROR on getting judgement  - Decision: " + idDecision + "/" + year)

    # get constitutionalParameters
    try:
        # visualize maxims
        visualizeMaximsButton = wd.find_element(By.XPATH, "//*[@id='filtro-massima']")
        wd.execute_script("arguments[0].click();", visualizeMaximsButton)

        labels = wd.find_elements(By.XPATH, "//*[contains(text(),'Parametri costituzionali')]")
        if len(labels) == 0:
            constitutional_parameters.append([""])
        else:
            for label in labels:
                param = []
                elem = wd.execute_script("""
                        return arguments[0].nextElementSibling
                    """, label.find_element(By.XPATH, ".."))  # get the element next to the div of the label
                while elem.tag_name != "br":
                    param.append(elem.text)  # append the constitutional parameter
                    elem = wd.execute_script("""
                        return arguments[0].nextElementSibling
                    """, elem)  # get the next element in the DOM
                constitutional_parameters.append(param)
    except NoSuchElementException:
        print("ERROR on getting constitutional parameters  - Decision: " + idDecision + "/" + year)
        constitutional_parameters.append([""])

    # close the bot
    wd.close()

    return judgement, constitutional_parameters


def fixExample(wrongExample):
    if wrongExample["codice_pronuncia"] in wrongExamples:
        judgement, constitutional_parameters = scrapeExample(wrongExample["codice_pronuncia"])
        wrongExample["judgement"] = judgement
        wrongExample["constitutional_parameters"] = constitutional_parameters
    return wrongExample

# fix wrong examples
wrongExamples = []  # specify 'codice_pronuncia' of wrong examples e.g ['110/1982']
updated_dataset = dataset.map(fixExample, load_from_cache_file=False)
print("\nUpdated dataset: ")
print(updated_dataset)

# visualize fixed examples
fixedExamples = updated_dataset.filter(lambda example: example["codice_pronuncia"] in wrongExamples)
print("\nFixed examples: ")
for example in fixedExamples:
    print(example["codice_pronuncia"])
    print(example["judgement"])
    print(example["constitutional_parameters"])

# save updated dataset
updated_dataset.save_to_disk(destinationPath + split)

dataset = load_from_disk(destinationPath + split)
print("\nUpdated dataset: ")
print(dataset)

# select 10 random examples
print("10 Random examples from updated dataset: ")
for _ in range(10):
    index = random.randint(0, len(dataset))
    print(dataset[index]["codice_pronuncia"])
    print(dataset[index]["judgement"])
    print(dataset[index]["constitutional_parameters"])

# examples with empty strings
emptyExamples = dataset.filter(lambda example: example["judgement"] == "")
print("Examples with no judgement: " + str(len(emptyExamples)))
