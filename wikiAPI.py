import requests
import os
import numpy as np
import logging

languages = ['en', 'sk', 'de', 'fr', 'it', 'cz', 'pl', 'hr', 'nl']
starting_article = 'Donald Trump'
n_articles = 10
folder_name = 'valid'

class attributes(object):
    def __init__(self):
        self.atts = {}
        self.atts['action'] = 'query'  # action=query
        self.atts['prop'] = 'extracts|links'  # prop=info
        self.atts['format'] = 'json'  # format=json
        # my_atts['titles'] = 'Freddie Mercury' # titles=Stanford%20University
        self.atts['explaintext'] = True
        self.atts['pllimit'] = 'max'
        self.atts['plnamespace'] = 0

    def set_title(self, title):
        self.atts['titles'] = title

    def reset(self):
        self.atts['prop'] = 'extracts|links'
        try:
            del self.atts['plcontinue']
        except KeyError:
            pass

    def set_links_only(self, pl):
        self.atts['prop'] = 'extracts|links'
        self.atts['plcontinue'] = pl

def send_request(title):
    my_atts.set_title(title)
    resp = requests.get(baseurl, params=my_atts.atts).json()
    return resp

def get_article_data(title):
    my_atts.reset()
    resp = send_request(title)
    try:
        text = [r['extract'] for r in resp['query']['pages'].values()][0]
        links = [link['title'] for link in [r['links'] for r in resp['query']['pages'].values()][0]]
    except KeyError:
        return None, None

    while True:
        try:
            pl = resp['continue']['plcontinue']
        except KeyError:
            break
        my_atts.set_links_only(pl)
        resp = send_request(title)
        links += [link['title'] for link in [r['links'] for r in resp['query']['pages'].values()][0]]

    return text, links

def save_text(dir_name, title, text, lang):
    with open(dir_name + '/' + lang + '/' + title.replace('/', '') + '.txt', 'w') as file:
        file.write(text)

def check_continue(title, n):
    global done, broken
    return title not in done and len(done) < n and title not in broken

def clean_string(s):
    return ''.join(ch for ch in s if (ch.isalnum() or ch == ' '))

def read_all_links(title, n, lang):
    if not check_continue(title, n):
        return
    text, links = get_article_data(title)
    if text == '' or text == None:
        broken.append(title)
        return
    save_text(folder_name, title, text, lang)

    global done
    logging.info(title + ' saved')
    done.append(title)

    np.random.shuffle(links)

    for l in links:
        read_all_links(l, n, lang)

def get_done_from_folder(dir_name):
    return [file[:-4] for file in os.listdir(dir_name)]

if __name__ == '__main__':


    logging.basicConfig(level=20)

    for lang in languages:
        logging.info('LANGUAGE: ' + lang)
        baseurl = 'http://' + lang + '.wikipedia.org/w/api.php'
        my_atts = attributes()
        if not os.path.exists(folder_name + '/'+lang):
            os.makedirs(folder_name + '/'+lang)

        done = []
        broken = []

        read_all_links(starting_article, n_articles, lang) #Stack Overflow

