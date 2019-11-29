# -*- coding: UTF-8 -*-
import os
import sys
import pymysql
import datetime
import json
import time
import glob
'''
/*
server:

10.200.128.156

*/
'''


def dump2file(filename, data_matrix, docid, sep='\t'):
	with open(filename, 'a') as f:
		for item in data_matrix:
			#urls = eval(str(item[0]))
			#url = []
			#for it in urls:
				#url.append(it['url'])
			#if item[1]==None:
				#continue
			data = json.loads(item[1])
			#if 'shortVideoType' in data and data['shortVideoType']==1 and 'shortVideoImage' in data:
			if 'picUrl' in data:
				mp4url = data['picUrl']
			#else:
			#	continue
			#item[0] = item[0].replace('\n',',')
				f.write('%s||%s\n' % (item[0], mp4url))


def get_article_vid(cur):
	#sql_rec = "select dbe.docid, dex.pic_url \
	#from  doc_base dbe left join doc_extra dex on dex.docid=dbe.docid where dbe.docid like 'd%' and \
	#dbe.publish_time<'2018-09-02' and dbe.publish_time>='2018-09-01'  and dbe.docType in (40, 43, 44, 45, 46, 48, 49, 11);"
	sql_rec = "select art0.docid, art0.body, art0.title, art0.tname \
	from  article_0 art0 where art0.update_time<'2018-11-22 12:20:00' and art0.update_time>='2018-11-22 12:00:00'"
	file_tp = '2022.txt'
	print('runing:')
	sql = sql_rec
	filesave = file_tp
	print(' > %s' % sql)
	selectRowNums = cur.execute(sql)
	selectResultList = cur.fetchall()
	dump2file(filesave, selectResultList)
	print('>>> %s' % filesave)

def get_article_by_docid(cur, docid):
	#sql_rec = "select  vi.urls, vi.title \
	#from  doc_base vi where vi.docid ='%s'" % (docid)	
	#sql_rec = "select db.category, db.sourceId, db.sourceLevel, db.interests \
#from doc_base db left join doc_extrainfo dx on db.docid=dx.docid where db.docid='%s'" %(docid)

	#sql_rec = "select db.docid, db.source, db.clusterId, db.sourceId from doc_base db where db.sourceId='%s' and db.publish_time>='2019-07-18 00:00:00'"%(docid)
	#sql_rec = "select db.title, db.category, ex.otherinfo from doc_base db join doc_extrainfo ex on db.docid=ex.docid where db.docid='%s'"%(docid)
        #sql_rec = "select db.docid, ex.otherinfo from doc_base db join doc_extrainfo ex on db.docid=ex.docid where db.docid='%s'"%(docid)
	#sql_rec = "select db.publish_time from doc_base db where db.docid='%s'"%(docid)
        sql_rec = "select db.category from doc_base db where db.docid='%s'"%(docid)
        #file_fp = 'video_category.txt'
        sql_rec1 = "select db.interests from doc_base db where db.docid='%s'"%(docid)
	#sql_rec2 = "select db.keywords from doc_base db where db.docid='%s'"%(docid)
        print('runing:')
	#sql = sql_rec
	#filesave = file_fp
        print(' > %s' % sql_rec)
        print(' > %s' % sql_rec1)
	#print(' > %s' % sql_rec2)
        cur.execute(sql_rec)
        selectResultList1 = cur.fetchall()
        cur.execute(sql_rec1)
        selectResultList2 = cur.fetchall()
	#cur.execute(sql_rec2)
	#selectResultList3 = cur.fetchall()
	#print(selectResultList1[0][0])
	#print(selectResultList2[0][0])
	#print(selectResultList3[0][0])
        with open('video_category.json','a+') as f:
            if(len(selectResultList1) > 0 and len(selectResultList1[0]) > 0):
                f.write(json.dumps((docid,selectResultList1[0][0]),ensure_ascii=False)+'\n')
        with open('video_interests.json','a+') as ff:
            if(len(selectResultList2) > 0 and len(selectResultList2[0][0]) > 0):
                ff.write(json.dumps((docid,selectResultList2[0][0]),ensure_ascii=False)+'\n')
	#with open('video_keywords.json','a+') as fff:
	#    if(type(selectResultList3) == None and len(selectResultList3) > 0 and len(selectResultList3[0][0]) > 0):
	#        fff.write(json.dumps((docid,selectResultList3[0][0]),ensure_ascii=False) + '\n')
	#dump2file(filesave, selectResultList, docid)
	#print('>>> %s' % filesave)

def main():
        conn=pymysql.connect(host='10.200.166.167',port=3306,user='bj_rec',passwd='OaiVXjGOs',db='contenthandle',charset='utf8')
        cur = conn.cursor()
        #get_article_vid(cur)
        #docid = 'VONPCG75E'
        #file_fp='get_imglist'
        with open('video_dict.txt','r') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            docid = line.split(':')[0]
            get_article_by_docid(cur, docid)
        conn.commit()
        cur.close()
        conn.close()

def test():
	conn=pymysql.connect(host='10.200.166.167',port=3306,user='bj_rec',passwd='OaiVXjGOs',db='contenthandle',charset='utf8')
	cur = conn.cursor()
	with open('/data/2/houxiaoxia/exin/list10-11.txt') as f:
		for line in f:
			line = line.strip()
			docid = line.split('\t')[1]
			get_article_by_docid(cur,docid)
	conn.commit()
	cur.close()
	conn.close()

def connect_abnormal():
	conn=pymysql.connect(host='106.38.231.10',port=3306,user='dy',passwd='oD2Eji9HAoqu',db='dy',charset='utf8')    
	cur = conn.cursor()
	sql_rec = "select dbe.docid, dex.pic_url \
        from  article_0 art0 where art0.updata_time<'2018-11-02 12:00:00' and art0.updata_time>='2018-11-02 12:05:00'"
	#sql = "SHOW TABLES"
	cur.execute(sql)
	result = cur.fetchall()
	results = []
	for i in range(len(result)):
		results.append(result[i])
	
	conn.close()
	print(results)

def abnormal():
	#conn=pymysql.connect(host='106.2.127.172',port=3306,user='bj_rec',passwd='OaiVXjGOs',db='contenthandle',charset='utf8')
	conn=pymysql.connect(host='10.200.131.144',port=4331,user='recsys_read',passwd='XYjBPSNkPdJ3',db='recsys',charset='utf8')
	cur = conn.cursor()
	print('cur',cur)
	#get_article_vid(cur)
	conn.commit()
	cur.close()
	conn.close()
if __name__ == '__main__':
	#abnormal()
        #test()
	main()
