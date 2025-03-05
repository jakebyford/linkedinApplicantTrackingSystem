# Import Dependencies
from selenium import webdriver
import time
from time import sleep
from bs4 import BeautifulSoup
import requests
import os
from config import my_directory

# create webdriver object
driver = webdriver.Chrome()


## Change ```keywords``` variable to search the job you want. 
## ```%20```` stands for whitespace

##### Web scraper for infinite scrolling page #####
keywords = 'data%20science'
# location = 'New%20Jersey%2C%20United%20States'
location = 'Florida%2C%20United%20States'
url = f'https://www.linkedin.com/jobs/search?keywords={keywords}&location={location}&geoId=&trk=public_jobs_jobs-search-bar_search-submit&position=1&pageNum=0&refresh=true'

print("===== Opening LinkedIn =====")
driver.get(url)
# time.sleep(2)  # Allow 2 seconds for the web page to open
scroll_pause_time = 2 # You can set your own pause time. My laptop is a bit slow so I use 1 sec
screen_height = driver.execute_script("return window.screen.height;")   # get the screen height of the web
i = 1

while True:
    # scroll one screen height each time
    driver.execute_script("window.scrollTo(0, {screen_height}*{i});".format(screen_height=screen_height, i=i))  
    i += 1
    time.sleep(scroll_pause_time)
    # update scroll height each time after scrolled, as the scroll height can change after we scrolled the page
    scroll_height = driver.execute_script("return document.body.scrollHeight;")  
    # Break the loop when the height we need to scroll to is larger than the total scroll height
    if (screen_height) * i > scroll_height:
        break 

##### Extract LinkedIn URLs #####
print("===== Extracting URLs =====")
urls = []
soup = BeautifulSoup(driver.page_source, "html.parser")
for link in soup.find_all('div', class_='base-card'):
    urls.append(link.a['href'])


## Directly change ```my_directory``` variable or 
# create a config.py file to import ```my_directory```

print("===== Saving URLs =====")
# my_directory = "C:/<example_Desktop>/<example_directory>"

os.chdir(my_directory)

SCRAPED_JOBS_PATH = my_directory + "/temp_data_science_job_descriptions"

SCRAPED_JOBS_PATH_EXIST = os.path.exists(SCRAPED_JOBS_PATH)

if SCRAPED_JOBS_PATH_EXIST == True:
    pass
else:
    os.mkdir(SCRAPED_JOBS_PATH)

## Loop through ```urls``` list to create 
## a job description file for each job

print("===== Scraping Job Descriptions =====")
for url in urls:
    page = requests.get(url)
    time.sleep(2)
    soup = BeautifulSoup(page.text, 'lxml')
    os.chdir(SCRAPED_JOBS_PATH)
    
    for div in soup.find_all('div', class_='top-card-layout__entity-info'):
        
        position_title = div.h1.text
        company = (div.h4.div.span.a.text).lstrip().rstrip()
        
        file_name = f"{position_title.lower().replace(' ', '_')}-{company.lower().replace(' ', '')}.txt"
        file_name = file_name.replace('(', "_") if file_name.__contains__("(") else file_name
        file_name = file_name.replace(')', "_") if file_name.__contains__(")") else file_name
        file_name_exists = os.path.exists(SCRAPED_JOBS_PATH + "/" + file_name)
        
        if file_name_exists:
            pass
        else:
            try:
                with open(file_name, "w", encoding="utf-8") as my_file:
                    section = soup.find_all('section', class_='show-more-less-html')
                    
                    section_divs = [i for i in section]
                    job_desc_sentences = [i.text for i in section_divs[0].find_all('li')]
                    job_desc_sentences = ' '.join(job_desc_sentences)
                        

                    try:
                        my_file.write(job_desc_sentences)
                        my_file.write("\n")

                    except FileNotFoundError:
                        pass
                        
#                     my_file.write("Job URL: " + url)
                    
            except (FileNotFoundError, OSError) as e:
                pass

print("===== Job Descriptions Scraped =====")
print("===== Closing LinkedIn =====")
driver.close()
