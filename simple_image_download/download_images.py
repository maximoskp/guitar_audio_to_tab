# from simple_image_download import Downloader 
import simple_image_download.simple_image_download as simp
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-k', '--keyword', type=str, default='cello-perform')
parser.add_argument('-n', '--num', type=int, default=60)
args = parser.parse_args()

my_downloader = simp.Downloader()


# Change Direcotory
my_downloader.directory = 'my_dir/'
# Change File extension type
my_downloader.extensions = '.jpg'
print(my_downloader.extensions)
my_downloader.download(args.keyword, limit=args.num, verbose=True)