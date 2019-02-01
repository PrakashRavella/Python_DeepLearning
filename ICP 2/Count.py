file1=open("read.txt", "r" )
words=0
char=0
line=file1.readline()
while(line!=""):
    words=words+len(line.split())
    char=char+len(line)-1
    print("words and lines are",words,char)
    words=0
    char=0
    line=file1.readline()
