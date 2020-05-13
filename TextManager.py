import glob, os
import sys

def FindFiles(extension):
    files = []
    os.chdir("./"+sys.argv[1])
    for file in glob.glob("*."+extension):
        files.append(file)
    return files


def ProcessFiles(files):
    for fileName in files:
        actualFile = open(fileName, encoding="utf8", errors="ignore")
        linesList = actualFile.readlines()
        finalList = []
        for line in linesList:
            line = line.rstrip()
            line = line.replace('<i>','')
            line = line.replace('</i>','')
            line = line.replace('<b>','')
            line = line.replace('</b>','')
            line = line.replace('<font>','')
            line = line.replace('</font>','')
            line = line.replace('♪','')
            line = line.replace('- ','')
            line = line.replace('"','')
            line = line.strip()
            if not line.isdigit() and not '-->' in line:
                finalList.append(line)
        finalList = finalList[4:]
        if not os.path.exists('NewSubs'):
            os.makedirs('NewSubs')
        with open('./NewSubs/' + fileName, 'w+', encoding="utf8", errors="ignore") as f:
            for item in finalList:
                if item is not '':
                    f.write("%s\n" % item)

# def ProcessInfo(files):
#     for fileName in files:
#         actualFile = open(fileName, encoding="utf8", errors="ignore")
#         linesList = actualFile.readlines()
#         finalList = []
#         for line in linesList:
#             if "FileName" in line:
#                 finalList.append(line)
#         if not os.path.exists('NewSubs'):
#             os.makedirs('NewSubs')
#         with open('./NewSubs/' + fileName, 'w+', encoding="utf8", errors="ignore") as f:
#             for item in finalList:
#                 if item is not '':
#                     f.write("%s\n" % item)

def Main():
    ProcessFiles(FindFiles("srt"))
    # ProcessInfo(FindFiles("nfo"))


Main()