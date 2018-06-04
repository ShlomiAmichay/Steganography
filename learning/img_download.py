from icrawler.builtin import GoogleImageCrawler
import base64
import threading
from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler
from six.moves.urllib.parse import urlparse


def my_crawl(name):
    class PrefixNameDownloader(ImageDownloader):
        def get_filename(self, task, default_ext):
            filename = super(PrefixNameDownloader, self).get_filename(task, default_ext)
            return name + filename

    google_crawler = GoogleImageCrawler(
        feeder_threads=1,
        parser_threads=2,
        downloader_cls=PrefixNameDownloader,
        downloader_threads=4,
        storage={'root_dir': '/Volumes/USB STICK/image database/images/google3'})
    filters = dict(
        size='=512x512',
        license='commercial,modify',
        date=((2017, 1, 1), (2017, 11, 30)))
    google_crawler.crawl(keyword=name , filters=filters, max_num=400, file_idx_offset=0)


f = open("keywords.txt", "r")
for line in f:
    my_crawl(line + " ")
