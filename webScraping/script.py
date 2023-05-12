import time
from selenium import webdriver
from selenium.common import NoSuchElementException
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.common.by import By
from datasets import load_from_disk, concatenate_datasets
import tqdm


def scrape(data, splitPath):
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

    # lists to populate new columns
    judgementList = []
    constitutionalParametersList = []

    for x in tqdm.tqdm(data):
        # split 'codice_pronuncia' in 'year' and 'idDecision'
        idDecision, year = x.split("/")

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
            print(
                "ERROR on " + splitPath + " visualizing the search result  - Decision: " + idDecision + "/" + year)
            judgementList.append("")
            constitutionalParametersList.append([[""]])
            # go back to the main page
            wd.get("https://www.cortecostituzionale.it/actionPronuncia.do")
            continue

        # get judgement
        try:
            judgement = wd.find_element(By.XPATH, "//*[@id='id1']/section[4]/div/div[1]/div[2]/strong").text
            judgementList.append(judgement)
        except NoSuchElementException:
            print("ERROR on " + splitPath + " getting judgement  - Decision: " + idDecision + "/" + year)
            judgementList.append("")

        # get constitutionalParameters
        try:
            # visualize maxims
            visualizeMaximsButton = wd.find_element(By.XPATH, "//*[@id='filtro-massima']")
            wd.execute_script("arguments[0].click();", visualizeMaximsButton)

            labels = wd.find_elements(By.XPATH, "//*[contains(text(),'Parametri costituzionali')]")
            if len(labels) == 0:
                constitutionalParametersList.append([[""]])
                # go back to the main page
                wd.get("https://www.cortecostituzionale.it/actionPronuncia.do")
                continue
            else:
                params = []
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
                    params.append(param)
                constitutionalParametersList.append(params)
        except NoSuchElementException:
            print(
                "ERROR on " + splitPath + " getting constitutional parameters  - Decision: " + idDecision + "/" + year)
            constitutionalParametersList.append([[""]])

        # go back to the main page
        wd.get("https://www.cortecostituzionale.it/actionPronuncia.do")

    # close the bot
    wd.close()

    return judgementList, constitutionalParametersList


def addColumnsToDataset(dataset, judgementList, constitutionalParametersList):
    dataset = dataset.add_column("judgement", judgementList)
    dataset = dataset.add_column("constitutional_parameters", constitutionalParametersList)
    return dataset


def saveDataset(dataset, splitPath):
    dataset.save_to_disk(destinationPath + splitPath)


def processShard(dataset, splitPath):
    # scraping web page
    judgementList, constitutionalParametersList = scrape(dataset["codice_pronuncia"], splitPath)
    # adding new columns to dataset
    dataset = addColumnsToDataset(dataset, judgementList, constitutionalParametersList)
    # saving dataset
    saveDataset(dataset, splitPath)


def execute(dataset, splitName, numShards):
    start = time.time()

    for i in range(numShards):
        shard = dataset.shard(num_shards=numShards, index=i)
        print("Scraping of " + splitName + "/" + "shard_" + str(i))
        processShard(shard, splitName + "/" + "shard_" + str(i))

    # concatenate shards
    dataset = load_from_disk(destinationPath + splitName + "/" + "shard_0")  # first shard
    for i in range(1, numShards):
        shard = load_from_disk(destinationPath + splitName + "/" + "shard_" + str(i))
        dataset = concatenate_datasets([dataset, shard])
    saveDataset(dataset, splitName)

    print(dataset)
    print("Elapsed time: " + str(time.time() - start))


def retryExecution(dataset, splitName, numShards, shardsWithError):
    start = time.time()
    numSubShards = 3

    for shardIndex in shardsWithError:
        shard = dataset.shard(num_shards=numShards, index=shardIndex)
        for subShardIndex in range(numSubShards):
            subShard = shard.shard(num_shards=numSubShards, index=subShardIndex)
            print("Scraping of " + splitName + "/" + "shard_" + str(shardIndex) + "/" + "subShard_" + str(
                subShardIndex))
            processShard(subShard, splitName + "/" + "shard_" + str(shardIndex) + "/" + "subShard_" + str(
                subShardIndex))
        # concatenate sub-shards
        shard = load_from_disk(destinationPath + splitName + "/" + "shard_" + str(
            shardIndex) + "/" + "subShard_0")  # first sub-shard
        for subShardIndex in range(1, numSubShards):
            subShard = load_from_disk(
                destinationPath + splitName + "/" + "shard_" + str(shardIndex) + "/" + "subShard_" + str(
                    subShardIndex))
            shard = concatenate_datasets([shard, subShard])
        saveDataset(shard, splitName + "/" + "shard_" + str(shardIndex))

    # concatenate all shards of the dataset, both old and recomputed ones
    dataset = load_from_disk(destinationPath + splitName + "/" + "shard_0")  # first shard
    for shardIndex in range(1, numShards):
        shard = load_from_disk(destinationPath + splitName + "/" + "shard_" + str(shardIndex))
        dataset = concatenate_datasets([dataset, shard])
    saveDataset(dataset, splitName)

    print(dataset)
    print("Elapsed time: " + str(time.time() - start))


sourcePath = "../datasets/LAWSU-IT/"
destinationPath = "../datasets/COMMA/"

# load dataset
lawsuit = load_from_disk(sourcePath)
train_dataset = lawsuit["train"]
eval_dataset = lawsuit["validation"]
test_dataset = lawsuit["test"]

# execute scraping and save result dataset
execute(train_dataset, "train", 20)
execute(eval_dataset, "validation", 3)
execute(test_dataset, "test", 3)

# re-scrape for the shards that had a lot of errors
shardsWithError = []  # specify the index of the shard, e.g [16] for the 15th shard
retryExecution(train_dataset, "train", 20, shardsWithError)
