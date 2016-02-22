import requests
import pandas as pd
import re
import ast
from bs4 import BeautifulSoup

if __name__ == '__main__': 

    URL_DEBATES = 'http://www.presidency.ucsb.edu/debates.php'
    SEPARATOR = '"undercard" debates'
    
    # Parse web page
    debatesPage = requests.get(URL_DEBATES)
    debatesSoup = BeautifulSoup(debatesPage.text, 'lxml')
    
    # Isolate debates table
    debatesTable = debatesSoup.find_all('table', {'width': '700', 'bgcolor': '#FFFFFF'})[0]
    
    # Isolate debates only for upcoming election
    relevantTable = debatesTable.prettify().split(SEPARATOR)[0]
    
    # Convert string back into a BeautifulSoup object for simple parsing
    df = pd.DataFrame()
    resoup = BeautifulSoup(relevantTable, 'lxml')
    tableRows = resoup.find('table').find_all('tr')
    for debate in tableRows:
        cells = debate.find_all('td')
        if len(cells) != 2:
            continue
        link = cells[1].find('a')
        if link:
            link = cells[1].a.get('href')
        else:
            continue
        date = cells[0].text.strip()
        title = cells[1].text.strip()
        data = {
            'Date': date,
            'Title': title,
            'Link': link
        }
        df = df.append(data, ignore_index=True)
    
    def scrape(df):    
        
        LINK = df['Link']
        print(LINK)
        page = requests.get(LINK)
        soup = BeautifulSoup(page.text, 'lxml')
        
        pageText = soup.find('span', {'class': 'displaytext'})
        pageText = re.sub(r'<b>(\.| |\n)*</b>', '', pageText.prettify()) # remove empty <b> tags
        pageText = pageText.split('<b>')
        
        partitions = []
        for partition in pageText:
            block = BeautifulSoup(partition, 'lxml')
            block = block.get_text().replace('\n', '')
            if block:
                partitions.append(block)
        
        # Regex pattern: match last word or last word before space+parenthesis
        pattern = re.compile(r'([a-zA-Z\']+)$|([a-zA-Z\']+)(?= \()') 
        
        # Identify participants
        participants = []
        participantsRaw = partitions.pop(0).split(';')
        for participant in participantsRaw:
            participant = pattern.search(participant)
            if participant:
                participants.append(participant.group().upper())
        
        # Consolidate all words for each participant
        master = {}
        for partition in partitions:
            partition = partition.strip()
            participant = re.search(r'(^\w+.*?|^\?.*?|^\[.*?):', partition).group(1)
            content = re.search(r'(?:^\w+.*?|^\?.*?|^\[.*?): *(.*)', partition).group(1)
            content = re.sub(r'\[.+?\]', '', content).strip() + " "
            
            if participant in participants and participant not in master:
                master[participant] = content
            elif participant in participants:
                master[participant] += content
        
        # Score each participant using nltk
        NLTK_API_URL = 'http://text-processing.com/api/sentiment/'
        scores = {}
        for key, value in master.items():
            response = requests.post(NLTK_API_URL, data={'text': value})
            responseJson = ast.literal_eval(response.content.decode('utf-8'))
            scores[key] = responseJson['probability']['pos']
            
        # Write each candidate's score and speech to dataframe column
        for candidate, score in scores.items():
            df[candidate] = score
        for candidate, speech in master.items():
            df[candidate+"_SPEECH"] = speech
        
        return df

    df = df.apply(scrape, axis=1)
    
    # Use a clean, ordered DataFrame for candidate scores without candidate speeches
    colList = [x for x in df.columns if '_SPEECH' not in x]
    firstColumns = ['Date', 'Link', 'Title']
    laterColumns = [x for x in colList if x not in firstColumns]
    colList = firstColumns + laterColumns
    
    data = df[colList]
    
    # Output dataset    
    data.to_csv("Debate sentiment.csv", index=False)    