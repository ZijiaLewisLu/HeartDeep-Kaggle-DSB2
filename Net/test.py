import time
now = time.ctime(int(time.time()))
print now
now = now.split(' ')
print now
#t = now[3].split(':')
#t = ':'.join(t[:2])
print '<' + now[-3] + '-' + now[-2] + '>'