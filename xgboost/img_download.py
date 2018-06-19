from icrawler import ImageDownloader
from icrawler.builtin import GoogleImageCrawler



def my_crawl(name):
    '''
    uses Google Image Crawler to crawl google image and download, according to given keyword
    :param name:
    :return:
    '''
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
    google_crawler.crawl(keyword=name + 'filetype: jpg' , filters=filters, max_num=500, file_idx_offset=0)


# opens keyword file and crawl google for all keywords on files
f = open("keywords.txt", "r")
for line in f:
    my_crawl(line + " ")
