import math
import re
import socket
import whois
import requests
import datetime
import urllib.parse as urlparse
from collections import Counter
from bs4 import BeautifulSoup

class FeatureExtraction:
    def __init__(self, url):
        self.url = url
        try:
            self.response = requests.get(url, timeout=5)
            self.soup = BeautifulSoup(self.response.text, "html.parser")
        except:
            self.response = None
            self.soup = None
    def using_ip(self):
        try:
            ip = socket.gethostbyname(urlparse.urlparse(self.url).netloc)
            return 1 if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", ip) else 0
        except:
            return 0

    def long_url(self):
        return 1 if len(self.url) > 75 else 0

    def short_url(self):
        shorteners = ["bit.ly", "goo.gl", "tinyurl", "t.co", "ow.ly"]
        return 1 if any(s in self.url for s in shorteners) else 0

    def symbol_at(self):
        return 1 if "@" in self.url else 0

    def redirecting(self):
        return 1 if "//" in self.url[urlparse.urlparse(self.url).scheme.__len__() + 2:] else 0

    def prefix_suffix(self):
        domain = urlparse.urlparse(self.url).netloc
        return 1 if "-" in domain else 0

    def sub_domains(self):
        domain = urlparse.urlparse(self.url).netloc.split(".")
        return 1 if len(domain) > 3 else 0

    def https(self):
        return 0 if self.url.startswith("https") else 1

    def domain_reg_len(self):
        try:
            domain = urlparse.urlparse(self.url).netloc
            w = whois.whois(domain)
            if w.expiration_date:
                exp_date = w.expiration_date
                if isinstance(exp_date, list):
                    exp_date = exp_date[0]
                days = (exp_date - datetime.datetime.now()).days
                return 1 if days / 365 <= 1 else 0
        except:
            return 1
        return 0

    def favicon(self):
        if self.soup:
            icon = self.soup.find("link", rel="icon")
            if icon and urlparse.urlparse(icon.get("href", "")).netloc not in self.url:
                return 1
        return 0

    def https_in_domain(self):
        domain = urlparse.urlparse(self.url).netloc
        return 1 if "https" in domain else 0

    def request_url(self):
        if self.soup:
            imgs = self.soup.find_all("img", src=True)
            total = len(imgs)
            external = sum(1 for img in imgs if urlparse.urlparse(img["src"]).netloc not in self.url)
            if total == 0:
                return 0
            return 1 if external / total > 0.5 else 0
        return 0

    def anchor_url(self):
        if self.soup:
            anchors = self.soup.find_all("a", href=True)
            total = len(anchors)
            external = sum(1 for a in anchors if urlparse.urlparse(a["href"]).netloc not in self.url)
            if total == 0:
                return 0
            return 1 if external / total > 0.5 else 0
        return 0

    def iframe_redirection(self):
        if self.soup:
            iframes = self.soup.find_all("iframe")
            return 1 if len(iframes) > 0 else 0
        return 0

    def age_of_domain(self):
        try:
            domain = urlparse.urlparse(self.url).netloc
            w = whois.whois(domain)
            if w.creation_date:
                creation_date = w.creation_date
                if isinstance(creation_date, list):
                    creation_date = creation_date[0]
                age_days = (datetime.datetime.now() - creation_date).days
                return 1 if age_days / 365 < 1 else 0
        except:
            return 1
        return 0

    def shannon_entropy(self):
        url_str = self.url.lower()
        prob = [count / len(url_str) for count in Counter(url_str).values()]
        entropy = -sum(p * math.log2(p) for p in prob)
        return 1 if entropy > 4.0 else 0
    def getFeaturesList(self):
        features = [
            self.using_ip(),
            self.long_url(),
            self.short_url(),
            self.symbol_at(),
            self.redirecting(),
            self.prefix_suffix(),
            self.sub_domains(),
            self.https(),
            self.domain_reg_len(),
            self.favicon(),
            self.https_in_domain(),
            self.request_url(),
            self.anchor_url(),
            self.iframe_redirection(),
            self.age_of_domain(),
            self.shannon_entropy()
        ]
        return features