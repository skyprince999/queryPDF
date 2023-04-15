import os
#import openai
import sklearn
from sklearn.metrics.pairwise import cosine_distances
import datetime
from collections import Counter
from time import time as now
import hashlib
import re
import io
import os
#import openai_functions as openai
from ai_bricks.api import openai
app_name='pdf qa'
__version__="2.0"
import streamlit as st
st.set_page_config(layout='centered', page_title=f'{app_name} {__version__}')
ss = st.session_state
os.environ["OPENAI_API_KEY"] = "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"
os.environ["COMMUNITY_USER"] = "Hemlata Channe"
STORAGE_MODE='LOCAL'
CACHE_MODE='DISK'

STORAGE_PATH='/home/lenovo/Documents/ThinkEvolve Projects/langchain project/qapdf/data/storage'
CACHE_PATH='/home/lenovo/Documents/ThinkEvolve Projects/langchain project/qapdf/data/cache'
os.chmod(CACHE_PATH,0o777)
v1 = """
/* feedback checkbox */
.css-18fuwiq {
 position: relative;
 padding-top: 6px;
}
.css-949r0i {
 position: relative;
 padding-top: 6px;
}
"""
# INFO: some prompts are still in model.py

# TODO: Ignore OCR problems in the text below.

TASK = {
	'v6': (
			"Answer the question truthfully based on the text below. "
			"Include verbatim quote and a comment where to find it in the text (page number). "
			#"After the quote write a step by step explanation in a new paragraph. "
			"After the quote write a step by step explanation. "
			"Use bullet points. "
			#"After that try to rephrase the original question so it might give better results. " 
		),
	'v5': (
			"Answer the question truthfully based on the text below. "
			"Include at least one verbatim quote (marked with quotation marks) and a comment where to find it in the text (ie name of the section and page number). "
			"Use ellipsis in the quote to omit irrelevant parts of the quote. "
			"After the quote write (in the new paragraph) a step by step explanation to be sure we have the right answer "
			"(use bullet-points in separate lines)" #, adjust the language for a young reader). "
			"After the explanation check if the Answer is consistent with the Context and doesn't require external knowledge. "
			"In a new line write 'SELF-CHECK OK' if the check was successful and 'SELF-CHECK FAILED' if it failed. " 
		),
	'v4':
		"Answer the question truthfully based on the text below. " \
		"Include verbatim quote and a comment where to find it in the text (ie name of the section and page number). " \
		"After the quote write an explanation (in the new paragraph) for a young reader.",
	'v3': 'Answer the question truthfully based on the text below. Include verbatim quote and a comment where to find it in the text (ie name of the section and page number).',
	'v2': 'Answer question based on context. The answers sould be elaborate and based only on the context.',
	'v1': 'Answer question based on context.',
	# 'v5':
		# "Generate a comprehensive and informative answer for a given question solely based on the provided document fragments. " \
		# "You must only use information from the provided fragments. Use an unbiased and journalistic tone. Combine fragments together into coherent answer. " \
		# "Do not repeat text. Cite fragments using [${number}] notation. Only cite the most relevant fragments that answer the question accurately. " \
		# "If different fragments refer to different entities with the same name, write separate answer for each entity.",
}

HYDE = "Write an example answer to the following question. Don't write generic answer, just assume everything that is not known."

# TODO
SUMMARY = {
	'v2':'Describe the document from which the fragment is extracted. Omit any details.',
	'v1':'Describe the document from which the fragment is extracted. Do not describe the fragment, focus on figuring out what kind document it is.',
}
"PDF adapter"

import pypdf
def clean_text(text):
	text = text.replace('\\','')
	text = text.replace('\n','')
	text = text.replace('\\x','')
	text = text.replace('\\xa','')
	text = text.replace('\\u201c','')
	
def pdf_to_pages(file):
	"extract text (pages) from pdf file"
	pages = []
	pdf = pypdf.PdfReader(file)
	for p in range(len(pdf.pages)):
		page = pdf.pages[p]
		text = page.extract_text()		
		pages += [text]
	return pages

import redis
from time import strftime
import os
from retry import retry

class Stats:
	def __init__(self):
		self.config = {}
	
	def render(self, key):
		variables = dict(
			date = strftime('%Y-%m-%d'),
			hour = strftime('%H'),
		)
		variables.update(self.config)
		for k,v in variables.items():
			key = key.replace('['+k+']',v)
		return key
	

class DictStats(Stats):
	def __init__(self, data_dict):
		self.data = data_dict
		self.config = {}
	
	def incr(self, key, kv_dict):
		data = self.data
		key = self.render(key)
		if key not in data:
			data[key] = {}
		for member,val in kv_dict.items():
			member = self.render(member)
			data[key][member] = data[key].get(member,0) + val
	
	def get(self, key):
		key = self.render(key)
		return self.data.get(key, {})


class RedisStats(Stats):
	def __init__(self):
		REDIS_URL = os.getenv('REDIS_URL')
		if not REDIS_URL:
			raise Exception('No Redis configuration in environment variables!')
		self.db = redis.Redis.from_url(REDIS_URL)
		self.config = {}
	
	@retry(tries=5, delay=0.1)
	def incr(self, key, kv_dict):
		# TODO: non critical code -> safe exceptions
		key = self.render(key)
		p = self.db.pipeline()
		for member,val in kv_dict.items():
			member = self.render(member)
			self.db.zincrby(key, val, member)
		p.execute()
	
	@retry(tries=5, delay=0.1)
	def get(self, key):
		# TODO: non critical code -> safe exceptions
		key = self.render(key)
		items = self.db.zscan_iter(key)
		return {k.decode('utf8'):v for k,v in items}


stats_data_dict = {}
def get_stats(**kw):
	MODE = os.getenv('STATS_MODE','').upper()
	if MODE=='REDIS':
		stats = RedisStats()
	else:
		stats = DictStats(stats_data_dict)
	stats.config.update(kw)
	return stats



if __name__=="__main__":
	s1 = get_stats(user='maciek')
	s1.incr('aaa:[date]:[user]', dict(a=1,b=2))
	s1.incr('aaa:[date]:[user]', dict(a=1,b=2))
	print(s1.data)
	print(s1.get('aaa:[date]:[user]'))
	#
	s2 = get_stats(user='kerbal')
	s2.incr('aaa:[date]:[user]', dict(a=1,b=2))
	s2.incr('aaa:[date]:[user]', dict(a=1,b=2))
	print(s2.data)
	print(s2.get('aaa:[date]:[user]'))
	DEFAULT_USER = os.getenv('COMMUNITY_USER','')

def use_key(key):
	openai.use_key(key)

usage_stats = get_stats(user=DEFAULT_USER)
def set_user(user):
	global usage_stats
	usage_stats = get_stats(user=user)
	#openai.set_global('user', user)
	#openai.add_callback('after', stats_callback)

def complete(text, **kw):
	model = kw.get('model','gpt-3.5-turbo')
	print("1.... model =",model)
	llm = openai.model(model)
	llm.config['pre_prompt'] = 'output only in raw text' # for chat models
	resp = llm.complete(text, **kw)
	resp['model'] = model
	return resp

def embedding(text, **kw):
	model = kw.get('model','text-embedding-ada-002')
    #model = 'text-embedding-ada-002'
	llm = openai.model(model)
	resp = llm.embed(text, **kw)
	#resp = openai.Embedding.create(input = [text],model = model)
	resp['model'] = model
	return resp

def embeddings(texts, **kw):
	model = kw.get('model','text-embedding-ada-002')
	#model = 'text-embedding-ada-002'
	llm = openai.model(model)
	resp = llm.embed_many(texts, **kw)	
	resp['model'] = model
	return resp

tokenizer_model = openai.model('text-davinci-003')

#tokenizer_model = ('text-davinci-003')
def get_token_count(text):
	return tokenizer_model.token_count(text)

def stats_callback(out, resp, self):
	model = self.config['model']
	usage = resp['usage']
	usage['call_cnt'] = 1
	if 'text' in out:
		usage['completion_chars'] = len(out['text'])
	elif 'texts' in out:
		usage['completion_chars'] = sum([len(text) for text in out['texts']])
	# TODO: prompt_chars
	# TODO: total_chars
	if 'rtt' in out:
		usage['rtt'] = out['rtt']
		usage['rtt_cnt'] = 1
	usage_stats.incr(f'usage:v4:[date]:[user]', {f'{k}:{model}':v for k,v in usage.items()})
	usage_stats.incr(f'hourly:v4:[date]',       {f'{k}:{model}:[hour]':v for k,v in usage.items()})
	#print('STATS_CALLBACK', usage, flush=True) # XXX

def get_community_usage_cost():
	data = usage_stats.get(f'usage:v4:[date]:{DEFAULT_USER}')
	used = 0.0
	used += 0.02   * data.get('total_tokens:text-davinci-003',0) / 1000
	used += 0.002  * data.get('total_tokens:text-curie-001',0) / 1000
	used += 0.002  * data.get('total_tokens:gpt-3.5-turbo',0) / 1000
	used += 0.0004 * data.get('total_tokens:text-embedding-ada-002',0) / 1000
	return used
#model
#def use_key(api_key):
	#use_key(api_key)

#def set_user(user):
	#set_user(user)

def query_by_vector(vector, index, limit=None):
	"return (ids, distances and texts) sorted by cosine distance"
	vectors = index['vectors']
	texts = index['texts']
	#
	sim = cosine_distances([vector], vectors)[0]
	#
	id_dist_list = list(enumerate(sim))
	id_dist_list.sort(key=lambda x:x[1])
	id_list   = [x[0] for x in id_dist_list][:limit]
	dist_list = [x[1] for x in id_dist_list][:limit]
	text_list = [texts[x] for x in id_list] if texts else ['ERROR']*len(id_list)
	return id_list, dist_list, text_list

def get_vectors(text_list):
	"transform texts into embedding vectors"
	batch_size = 128
	vectors = []
	usage = Counter()
	for i,texts in enumerate(batch(text_list, batch_size)):
		resp = embeddings(texts)
		v = resp['vectors']
		u = resp['usage']
		u['call_cnt'] = 1
		usage.update(u)
		vectors.extend(v)
	return {'vectors':vectors, 'usage':dict(usage), 'model':resp['model']}

def index_file(f, filename, fix_text=False, frag_size=0, cache=None):
	"return vector index (dictionary) for a given PDF file"
	# calc md5
	h = hashlib.md5()
	h.update(f.read())
	md5 = h.hexdigest()
	filesize = f.tell()
	f.seek(0)
	#
	t0 = now()
	pages = pdf_to_pages(f)
	t1 = now()
	
	if fix_text:
		for i in range(len(pages)):
			pages[i] = fix_text_problems(pages[i])
	texts = split_pages_into_fragments(pages, frag_size)
	t2 = now()
	if cache:
		cache_key = f'get_vectors:{md5}:{frag_size}:{fix_text}'
		resp = cache.call(cache_key, get_vectors, texts)
	else:
		resp = get_vectors(texts)
	
	t3 = now()
	vectors = resp['vectors']
	summary_prompt = f"{texts[0]}\n\nDescribe the document from which the fragment is extracted. Omit any details.\n\n" # TODO: move to prompts.py
	summary = complete(summary_prompt)
	t4 = now()
	usage = resp['usage']
	out = {}
	out['frag_size'] = frag_size
	out['n_pages']   = len(pages)
	out['n_texts']   = len(texts)
	out['texts']     = texts
	out['pages']     = pages
	out['vectors']   = vectors
	out['summary']   = summary['text']
	out['filename']  = filename
	out['filehash']  = f'md5:{md5}'
	out['filesize']  = filesize
	out['usage']     = usage
	out['model']     = resp['model']
	out['time']      = {'pdf_to_pages':t1-t0, 'split_pages':t2-t1, 'get_vectors':t3-t2, 'summary':t4-t3}
	out['size']      = len(texts)   # DEPRECATED -> filesize
	out['hash']      = f'md5:{md5}' # DEPRECATED -> filehash
	return out

def split_pages_into_fragments(pages, frag_size):
	"split pages (list of texts) into smaller fragments (list of texts)"
	page_offset = [0]
	for p,page in enumerate(pages):
		page_offset += [page_offset[-1]+len(page)+1]
	# TODO: del page_offset[-1] ???
	if frag_size:
		text = ' '.join(pages)
		return text_to_fragments(text, frag_size, page_offset)
	else:
		return pages

def text_to_fragments(text, size, page_offset):
	"split single text into smaller fragments (list of texts)"
	if size and len(text)>size:
		out = []
		pos = 0
		page = 1
		p_off = page_offset.copy()[1:]
		eos = find_eos(text)
		if len(text) not in eos:
			eos += [len(text)]
		for i in range(len(eos)):
			if eos[i]-pos>size:
				text_fragment = f'PAGE({page}):\n'+text[pos:eos[i]]
				out += [text_fragment]
				pos = eos[i]
				if eos[i]>p_off[0]:
					page += 1
					del p_off[0]
		# ugly: last iter
		text_fragment = f'PAGE({page}):\n'+text[pos:eos[i]]
		out += [text_fragment]
		#
		out = [x for x in out if x]
		return out
	else:
		return [text]

def find_eos(text):
	"return list of all end-of-sentence offsets"
	return [x.span()[1] for x in re.finditer('[.!?]\s+',text)]

###############################################################################

def fix_text_problems(text):
	"fix common text problems"
	text = re.sub('\s+[-]\s+','',text) # word continuation in the next line
	print("type ",type(text))
	#text = re.sub('\\','',text)    #clean_text(text)
	return text

def query(text, index, task=None, temperature=0.0, max_frags=1, hyde=False, hyde_prompt=None, limit=None, n_before=1, n_after=1, model=None):
	"get dictionary with the answer for the given question (text)."
	out = {}
	
	if hyde:
		# TODO: model param
		out['hyde'] = hypotetical_answer(text, index, hyde_prompt=hyde_prompt, temperature=temperature)
		# TODO: usage
	
	# RANK FRAGMENTS
	if hyde:
		resp = embedding(out['hyde']['text'])
		# TODO: usage
	else:
		resp = embedding(text)
		# TODO: usage
	v = resp['vector']
	t0 = now()
	id_list, dist_list, text_list = query_by_vector(v, index, limit=limit)
	dt0 = now()-t0
	
	# BUILD PROMPT
	
	# select fragments
	N_BEFORE = 1 # TODO: param
	N_AFTER =  1 # TODO: param
	selected = {} # text id -> rank
	for rank,id in enumerate(id_list):
		for x in range(id-n_before, id+1+n_after):
			if x not in selected and x>=0 and x<index['size']:
				selected[x] = rank
	selected2 = [(id,rank) for id,rank in selected.items()]
	selected2.sort(key=lambda x:(x[1],x[0]))
	
	# build context
	SEPARATOR = '\n---\n'
	context = ''
	context_len = 0
	frag_list = []
	for id,rank in selected2:
		frag = index['texts'][id]
		frag_len = get_token_count(frag)
		if context_len+frag_len <= 3000: # TODO: remove hardcode
			context += SEPARATOR + frag # add separator and text fragment
			frag_list += [frag]
			context_len = get_token_count(context)
	out['context_len'] = context_len
	prompt = f"""
		{task or 'Task: Answer question based on context.'}
		
		Context:
		{context}
		
		Question: {text}
		
		Answer:""" # TODO: move to prompts.py
	
	# GET ANSWER
	resp2 = complete(prompt, temperature=temperature, model=model)
	answer = resp2['text']
	usage = resp2['usage']
	
	# OUTPUT
	out['vector_query_time'] = dt0
	out['id_list'] = id_list
	out['dist_list'] = dist_list
	out['selected'] = selected
	out['selected2'] = selected2
	out['frag_list'] = frag_list
	#out['query.vector'] = resp['vector']
	out['usage'] = usage
	out['prompt'] = prompt
	out['model'] = resp2['model']
	# CORE
	out['text'] = answer
	return out

def hypotetical_answer(text, index, hyde_prompt=None, temperature=0.0):
	"get hypotethical answer for the question (text)"
	hyde_prompt = hyde_prompt or 'Write document that answers the question.'
	prompt = f"""
	{hyde_prompt}
	Question: "{text}"
	Document:""" # TODO: move to prompts.py
	resp = complete(prompt, temperature=temperature)
	return resp


def community_tokens_available_pct():
	used = get_community_usage_cost()
	limit = float(os.getenv('COMMUNITY_DAILY_USD',0))
	pct = (100.0 * (limit-used) / limit) if limit else 0
	pct = max(0, pct)
	pct = min(100, pct)
	return pct


def community_tokens_refresh_in():
	x = datetime.datetime.now()
	dt = (x.replace(hour=23, minute=59, second=59) - x).seconds
	h = dt // 3600
	m = dt  % 3600 // 60
	return f"{h} h {m} min"

# util
def batch(data, n):
	for i in range(0, len(data), n):
		yield data[i:i+n]

if __name__=="__main__":
	print(text_to_fragments("to test. test 2. program", size=3, page_offset=[0,5,10,15,20]))
	
"Storage adapter - one folder for each user / api_key"

# pip install pycryptodome
# REF: https://www.pycryptodome.org/src/cipher/aes
import Crypto
from Crypto.Cipher import AES
from Crypto.Util.Padding import pad,unpad

from retry import retry

from binascii import hexlify,unhexlify
import hashlib
import pickle
import zlib
import os
import io

# pip install boto3
import boto3
import botocore

SALT = unhexlify(os.getenv('STORAGE_SALT','00'))

class Storage:
	"Encrypted object storage (base class)"
	
	def __init__(self, secret_key):
		k = secret_key.encode()
		self.folder = hashlib.blake2s(k, salt=SALT, person=b'folder', digest_size=8).hexdigest()
		self.passwd = hashlib.blake2s(k, salt=SALT, person=b'passwd', digest_size=32).hexdigest()
		self.AES_MODE = AES.MODE_ECB # TODO: better AES mode ???
		self.AES_BLOCK_SIZE = 16
	
	def get(self, name, default=None):
		"get one object from the folder"
		safe_name = self.encode(name)
		data = self._get(safe_name)
		obj = self.deserialize(data)
		return obj
	
	def put(self, name, obj):
		"put the object into the folder"
		safe_name = self.encode(name)
		data = self.serialize(obj)
		self._put(safe_name, data)
		return data

	def list(self):
		"list object names from the folder"
		return [self.decode(name) for name in self._list()]

	def delete(self, name):
		"delete the object from the folder"
		safe_name = self.encode(name)
		self._delete(safe_name)
	
	# IMPLEMENTED IN SUBCLASSES
	def _put(self, name, data):
		...
	def _get(self, name):
		...	
	def _delete(self, name):
		pass
	def _list(self):
		...
	
	# # #
	
	def serialize(self, obj):
		raw = pickle.dumps(obj)
		compressed = self.compress(raw)
		encrypted = self.encrypt(compressed)
		return encrypted
	
	def deserialize(self, encrypted):
		compressed = self.decrypt(encrypted)
		raw = self.decompress(compressed)
		obj = pickle.loads(raw)
		return obj

	def encrypt(self, raw):
		cipher = AES.new(unhexlify(self.passwd), self.AES_MODE)
		return cipher.encrypt(pad(raw, self.AES_BLOCK_SIZE))
	
	def decrypt(self, encrypted):
		cipher = AES.new(unhexlify(self.passwd), self.AES_MODE)
		return unpad(cipher.decrypt(encrypted), self.AES_BLOCK_SIZE)

	def compress(self, data):
		return zlib.compress(data)
	
	def decompress(self, data):
		return zlib.decompress(data)
	
	def encode(self, name):
		return hexlify(name.encode('utf8')).decode('utf8')
	
	def decode(self, name):
		return unhexlify(name).decode('utf8')


class DictStorage(Storage):
	"Dictionary based storage"
	
	def __init__(self, secret_key, data_dict):
		super().__init__(secret_key)
		self.data = data_dict
		
	def _put(self, name, data):
		if self.folder not in self.data:
			self.data[self.folder] = {}
		self.data[self.folder][name] = data
		
	def _get(self, name):
		return self.data[self.folder][name]
	
	def _list(self):
		# TODO: sort by modification time (reverse=True)
		return list(self.data.get(self.folder,{}).keys())
	
	def _delete(self, name):
		del self.data[self.folder][name]


class LocalStorage(Storage):
	"Local filesystem based storage"
	
	def __init__(self, secret_key, path):
		if not path:
			raise Exception('No storage path in environment variables!')
		super().__init__(secret_key)
		self.path = os.path.join(path, self.folder)
		if not os.path.exists(self.path):
			os.makedirs(self.path)
	
	def _put(self, name, data):
		with open(os.path.join(self.path, name), 'wb') as f:
			f.write(data)

	def _get(self, name):
		with open(os.path.join(self.path, name), 'rb') as f:
			data = f.read()
		return data
	
	def _list(self):
		# TODO: sort by modification time (reverse=True)
		return os.listdir(self.path)
	
	def _delete(self, name):
		os.remove(os.path.join(self.path, name))


class S3Storage(Storage):
	"S3 based encrypted storage"
	
	def __init__(self, secret_key, **kw):
		prefix = kw.get('prefix') or os.getenv('S3_PREFIX','index/x1')
		region = kw.get('region') or os.getenv('S3_REGION','sfo3')
		bucket = kw.get('bucket') or os.getenv('S3_BUCKET','ask-my-pdf')
		url    = kw.get('url')    or os.getenv('S3_URL',f'https://{region}.digitaloceanspaces.com')
		key    = os.getenv('S3_KEY','')
		secret = os.getenv('S3_SECRET','')
		#
		if not key or not secret:
			raise Exception("No S3 credentials in environment variables!")
		#
		super().__init__(secret_key)
		self.session = boto3.session.Session()
		self.s3 = self.session.client('s3',
				config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
				region_name=region,
				endpoint_url=url,
				aws_access_key_id=key,
				aws_secret_access_key=secret,
			)
		self.bucket = bucket
		self.prefix = prefix
	
	def get_key(self, name):
		return f'{self.prefix}/{self.folder}/{name}'
	
	def _put(self, name, data):
		key = self.get_key(name)
		f = io.BytesIO(data)
		self.s3.upload_fileobj(f, self.bucket, key)
	
	def _get(self, name):
		key = self.get_key(name)
		f = io.BytesIO()
		self.s3.download_fileobj(self.bucket, key, f)
		f.seek(0)
		return f.read()
	
	def _list(self):
		resp = self.s3.list_objects(
				Bucket=self.bucket,
				Prefix=self.get_key('')
			)
		contents = resp.get('Contents',[])
		contents.sort(key=lambda x:x['LastModified'], reverse=True)
		keys = [x['Key'] for x in contents]
		names = [x.split('/')[-1] for x in keys]
		return names
	
	def _delete(self, name):
		self.s3.delete_object(
				Bucket=self.bucket,
				Key=self.get_key(name)
			)

def get_storage(api_key, data_dict):
	"get storage adapter configured in environment variables"
	mode = os.getenv('STORAGE_MODE','').upper()
	path = os.getenv('STORAGE_PATH','')
	if mode=='S3':
		storage = S3Storage(api_key)
	elif mode=='LOCAL':
		storage = LocalStorage(api_key, path)
	else:
		storage = DictStorage(api_key, data_dict)
	return storage


#cache
from retry import retry

from binascii import hexlify,unhexlify
import pickle
import zlib
import io
import os

# pip install boto3
import boto3
import botocore

class Cache:
	"Dummy / Base Cache"
	def __init__(self):
		pass
	
	def put(self, key, obj):
		pass
	
	def get(self, key):
		return None
	
	def has(self, key):
		return False
	
	def delete(self, key):
		pass

	def serialize(self, obj):
		pickled = pickle.dumps(obj)
		compressed = self.compress(pickled)
		return compressed
	
	def deserialize(self, data):
		pickled = self.decompress(data)
		obj = pickle.loads(pickled)
		return obj

	def compress(self, data):
		return zlib.compress(data)
	
	def decompress(self, data):
		print("data",data)
		return zlib.decompress(data)

	def encode(self, name):
		return hexlify(name.encode('utf8')).decode('utf8')
	
	def decode(self, name):
		return unhexlify(name).decode('utf8')
	
	def call(self, key, fun, *a, **kw):
		if self.has(key):
			return self.get(key)
		else:
			resp = fun(*a, **kw)
			self.put(key, resp)
			return resp


class DiskCache(Cache):
	"Local disk based cache"

	def __init__(self, root):
		self.root = root
	
	def path(self, key):
		return os.path.join(self.root, self.encode(key))	
	
	def put(self, key, obj):
		path = self.path(key)
		print("1.....path ",path)
		data = self.serialize(obj)
		if os.path.exists(path):
			os.chmod(path,0o777)
		with open(path, 'wb') as f:
			f.write(data)
			f.close()
		    #f.close()
		    
	def get(self,key):
		path = self.path(key)
		#data = "hello"
		with open(path,"rb") as f:
			data = f.read()
			obj = self.deserialize(data)
			
		return obj
		
    

	def has(self, key):
		path = self.path(key)
		return os.path.exists(path)

	def delete(self, key):
		path = self.path(key)
		os.remove(path)


class S3Cache(Cache):
	"S3 based cache"

	def __init__(self, **kw):
		bucket = kw.get('bucket') or os.getenv('S3_CACHE_BUCKET','ask-my-pdf')
		prefix = kw.get('prefix') or os.getenv('S3_CACHE_PREFIX','cache/x1')
		region = kw.get('region') or os.getenv('S3_REGION','sfo3')
		url    = kw.get('url')    or os.getenv('S3_URL',f'https://{region}.digitaloceanspaces.com')
		key    = os.getenv('S3_KEY','')
		secret = os.getenv('S3_SECRET','')
		#
		if not key or not secret:
			raise Exception("No S3 credentials in environment variables!")
		#
		self.session = boto3.session.Session()
		self.s3 = self.session.client('s3',
				config=botocore.config.Config(s3={'addressing_style': 'virtual'}),
				region_name=region,
				endpoint_url=url,
				aws_access_key_id=key,
				aws_secret_access_key=secret,
			)
		self.bucket = bucket
		self.prefix = prefix

	def get_s3_key(self, key):
		return f'{self.prefix}/{key}'
	
	def put(self, key, obj):
		s3_key = self.get_s3_key(key)
		data = self.serialize(obj)
		f = io.BytesIO(data)
		self.s3.upload_fileobj(f, self.bucket, s3_key)

	def get(self, key, default=None):
		s3_key = self.get_s3_key(key)
		f = io.BytesIO()
		try:
			self.s3.download_fileobj(self.bucket, s3_key, f)
		except:
			f.close()
			return default
		f.seek(0)
		data = f.read()
		obj = self.deserialize(data)
		return obj
	
	def has(self, key):
		s3_key = self.get_s3_key(key)
		try:
			self.s3.head_object(Bucket=self.bucket, Key=s3_key)
			return True
		except:
			return False
	
	def delete(self, key):
		self.s3.delete_object(
			Bucket = self.bucket,
			Key = self.get_s3_key(key))


def get_cache(**kw):
	mode = CACHE_MODE   #os.getenv('CACHE_MODE','').upper()
	path = CACHE_PATH #os.getenv('CACHE_PATH','').upper()
	os.chmod(CACHE_PATH,0o777)
	print("path ",path)
	if mode == 'DISK':
		return DiskCache(path)
	elif mode == 'S3':
		return S3Cache(**kw)
	else:
		return Cache()

import subprocess

#subprocess.run(['chmod', '0777', '/data'])
if __name__=="__main__":
    cache = DiskCache(CACHE_PATH)
    #os.chmod('/data', 0o777)
    #cache = DiskCache(cache)
	
    cache.put('xxx',{'a':1,'b':22})
    print('get xxx', cache.get('xxx'))
    print('has xxx', cache.has('xxx'))
    print('has yyy', cache.has('yyy'))
    #print('delete xxx', cache.delete('xxx'))
    print('has xxx', cache.has('xxx'))
    print('get xxx', cache.get('xxx'))
	
import datetime
import hashlib
import redis
import os
from retry import retry

def hexdigest(text):
	return hashlib.md5(text.encode('utf8')).hexdigest()

def as_int(x):
	return int(x) if x is not None else None

class Feedback:
	"Dummy feedback adapter"
	def __init__(self, user):
		...
	def send(self, score, ctx, details=False):
		...
	def get_score(self):
		return 0

class RedisFeedback(Feedback):
	"Redis feedback adapter"
	def __init__(self, user):
		REDIS_URL = os.getenv('REDIS_URL')
		if not REDIS_URL:
			raise Exception('No Redis configuration in environment variables!')
		super().__init__(user)
		self.db = redis.Redis.from_url(REDIS_URL)
		self.user = user

	@retry(tries=5, delay=0.1)
	def send(self, score, ctx, details=False):
		p = self.db.pipeline()
		dist_list = ctx.get('debug',{}).get('model.query.resp',{}).get('dist_list',[])
		# feedback
		index = ctx.get('index',{})
		data = {}
		data['user'] = self.user
		data['task-prompt-version'] = ctx.get('task_name')
		data['model'] = ctx.get('model')
		data['model-embeddings'] = ctx.get('model_embed')
		data['task-prompt'] = ctx.get('task')
		data['temperature'] = ctx.get('temperature')
		data['frag-size'] = ctx.get('frag_size')
		data['frag-cnt'] = ctx.get('max_frags')
		data['frag-n-before'] = ctx.get('n_frag_before')
		data['frag-n-after'] = ctx.get('n_frag_after')
		data['filename'] = ctx.get('filename')
		data['filehash'] = index.get('hash') or index.get('filehash')
		data['filesize'] = index.get('filesize')
		data['n-pages'] = index.get('n_pages')
		data['n-texts'] = index.get('n_texts')
		data['use-hyde'] = as_int(ctx.get('use_hyde'))
		data['use-hyde-summary'] = as_int(ctx.get('use_hyde_summary'))
		data['question'] = ctx.get('question')
		data['answer'] = ctx.get('answer')
		data['hyde-summary'] = index.get('summary')
		data['resp-dist-list'] = '|'.join([f"{x:0.3f}" for x in dist_list])
		fb_hash = hexdigest(str(list(sorted(data.items()))))
		#
		data['score'] = score
		data['datetime'] = str(datetime.datetime.now())
		key1 = f'feedback:v2:{fb_hash}'
		if not details:
			for k in ['question','answer','hyde-summary']:
				data[k] = ''
		p.hset(key1, mapping=data)
		# feedback-daily
		date = datetime.date.today()
		key2 = f'feedback-daily:v1:{date}:{"positive" if score > 0 else "negative"}'
		p.sadd(key2, fb_hash)
		# feedback-score
		key3 = f'feedback-score:v2:{self.user}'
		p.sadd(key3, fb_hash)
		p.execute()
	
	@retry(tries=5, delay=0.1)
	def get_score(self):
		key = f'feedback-score:v2:{self.user}'
		return self.db.scard(key)


def get_feedback_adapter(user):
	MODE = os.getenv('FEEDBACK_MODE','').upper()
	if MODE=='REDIS':
		return RedisFeedback(user)
	else:
		return Feedback(user)

if 'debug' not in ss: ss['debug'] = {}

st.write(f'<style>{v1}</style>', unsafe_allow_html=True)
header1 = st.empty() # for errors / messages
header2 = st.empty() # for errors / messages
header3 = st.empty() # for errors / messages



# HANDLERS

def on_api_key_change():
  api_key = os.getenv('OPENAI_API_KEY')
  os.environ["OPENAI_API_KEY"] = "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"
  use_key(api_key) # TODO: empty api_key
  openai.api_key = "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"
	#
  api_key = "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"
  if 'task_name' not in st.session_state:
    st.session_state['task_name']= 'V6'
  if 'user' not in st.session_state:
    st.session_state['user']= 'hemlata'
  if 'data_dict' not in ss: 
    ss['data_dict'] = {} # used only with DictStorage
  ss['storage'] = get_storage(api_key, data_dict=ss['data_dict'])
  ss['cache'] = get_cache()
  ss['user'] = ss['storage'].folder # TODO: refactor user 'calculation' from get_storage
  set_user(ss['user'])
  ss['feedback'] = get_feedback_adapter(ss['user'])
  ss['feedback_score'] = ss['feedback'].get_score()
	#
  ss['debug']['storage.folder'] = ss['storage'].folder
  ss['debug']['storage.class'] = ss['storage'].__class__.__name__
  ss['task_name'] = 'v6'
  ss['user'] = 'hemlata'
  #ss['user'] = 'hemlata' #os.getenv('COMMUNITY_USER')
  #if 'user' in ss:
#on_api_key_change() # use community key

# COMPONENTS


def ui_spacer(n=2, line=False, next_n=0):
	for _ in range(n):
		st.write('')
	if line:
		st.tabs([' '])
	for _ in range(next_n):
		st.write('')

def ui_info():
	st.markdown(f"""
	# Ask my PDF
	version {__version__}
	
	Question answering system built on top of GPT3.
	""")
	ui_spacer(1)
	st.write("Made by ------Name", unsafe_allow_html=True)
	ui_spacer(1)
	st.markdown("""
		QA on pdf document
		""")
	ui_spacer(1)
	st.markdown('hello.')

def ui_api_key():
    st.write('## 1. Enter your OpenAI API key')
    st.text_input('OpenAI API key', type='password', key='api_key', on_change=on_api_key_change, label_visibility="collapsed")

def index_pdf_file():
	if ss['pdf_file']:
		ss['filename'] = ss['pdf_file'].name
		if ss['filename'] != ss.get('fielname_done'): 
			with st.spinner(f'indexing {ss["filename"]}'):
				print("cache ",ss['cache'])
				index = index_file(ss['pdf_file'], ss['filename'], fix_text=ss['fix_text'], frag_size=ss['frag_size'], cache=ss['cache'])
				ss['index'] = index
				debug_index()
				ss['filename_done'] = ss['filename'] 

def debug_index():
	index = ss['index']
	d = {}
	d['hash'] = index['hash']
	d['frag_size'] = index['frag_size']
	d['n_pages'] = len(index['pages'])
	d['n_texts'] = len(index['texts'])
	d['summary'] = index['summary']
	d['pages'] = index['pages']
	d['texts'] = index['texts']
	d['time'] = index.get('time',{})
	ss['debug']['index'] = d

def ui_pdf_file():
	st.write('## 2. Upload or select your PDF file')
	disabled = not ss.get('user') or (not ss.get('api_key') )  #and not ss.get('community_pct',0))
	t1,t2 = st.tabs(['UPLOAD','SELECT'])
	with t1:
		st.file_uploader('pdf file', type='pdf', key='pdf_file', disabled=disabled, on_change=index_pdf_file, label_visibility="collapsed")
		b_save()
	with t2:
		filenames = ['']
		if ss.get('storage'):
			filenames += ss['storage'].list()
		def on_change():
			name = ss['selected_file']
			if name and ss.get('storage'):
				with ss['spin_select_file']:
					with st.spinner('loading index'):
						t0 = now()
						index = ss['storage'].get(name)
						ss['debug']['storage_get_time'] = now()-t0
				ss['filename'] = name # XXX
				ss['index'] = index
				debug_index()
			else:
				#ss['index'] = {}
				pass
		st.selectbox('select file', filenames, on_change=on_change, key='selected_file', label_visibility="collapsed", disabled=disabled)
		b_delete()
		ss['spin_select_file'] = st.empty()

def ui_show_debug():
	st.checkbox('show debug section', key='show_debug')

def ui_fix_text():
	st.checkbox('fix common PDF problems', value=True, key='fix_text')

def ui_temperature():
	#st.slider('temperature', 0.0, 1.0, 0.0, 0.1, key='temperature', format='%0.1f')
	ss['temperature'] = 0.0

def ui_fragments():
	#st.number_input('fragment size', 0,2000,200, step=100, key='frag_size')
	st.selectbox('fragment size (characters)', [0,200,300,400,500,600,700,800,900,1000], index=3, key='frag_size')
	b_reindex()
	st.number_input('max fragments', 1, 10, 4, key='max_frags')
	st.number_input('fragments before', 0, 3, 1, key='n_frag_before') # TODO: pass to model
	st.number_input('fragments after',  0, 3, 1, key='n_frag_after')  # TODO: pass to model

def ui_model():
	models = ['gpt-3.5-turbo','text-davinci-003','text-curie-001']
	st.selectbox('main model', models, key='model', disabled=not ss.get('api_key'))
	st.selectbox('embedding model', ['text-embedding-ada-002'], key='model_embed') # FOR FUTURE USE

def ui_hyde():
	st.checkbox('use HyDE', value=True, key='use_hyde')

def ui_hyde_summary():
	st.checkbox('use summary in HyDE', value=True, key='use_hyde_summary')

def ui_task_template():
	st.selectbox('task prompt template', TASK.keys(), key='task_name')

def ui_task():
	x = 'v6'#st['task_name']
	st.text_area('task prompt', TASK[x], key='task')

def ui_hyde_prompt():
	st.text_area('HyDE prompt', HYDE, key='hyde_prompt')

def ui_question():
	st.write('## 3. Ask questions'+(f' to {ss["filename"]}' if ss.get('filename') else ''))
	disabled = False
	st.text_area('question', key='question', height=100, placeholder='Enter question here', help='', label_visibility="collapsed", disabled=disabled)

# REF: Hypotetical Document Embeddings
def ui_hyde_answer():
	# TODO: enter or generate
	pass

def ui_output():
	output = ss.get('output','')
	st.markdown(output)

def ui_debug():
	if ss.get('show_debug'):
		st.write('### debug')
		st.write(ss.get('debug',{}))


def b_ask():
	c1,c2,c3,c4,c5 = st.columns([2,1,1,2,2])
	if c2.button('👍', use_container_width=True, disabled=not ss.get('output')):
		ss['feedback'].send(+1, ss, details=ss['send_details'])
		ss['feedback_score'] = ss['feedback'].get_score()
	if c3.button('👎', use_container_width=True, disabled=not ss.get('output')):
		ss['feedback'].send(-1, ss, details=ss['send_details'])
		ss['feedback_score'] = ss['feedback'].get_score()
	score = ss.get('feedback_score',0)
	c5.write(f'feedback score: {score}')
	c4.checkbox('send details', True, key='send_details',
			help='allow question and the answer to be stored in the ask-my-pdf feedback database')
	#c1,c2,c3 = st.columns([1,3,1])
	#c2.radio('zzz',['👍',r'...',r'👎'],horizontal=True,label_visibility="collapsed")
	#
	disabled = (not ss.get('api_key') and not ss.get('community_pct',0)) or not ss.get('index')
	if c1.button('get answer', disabled=disabled, type='primary', use_container_width=True):
		question = ss.get('question','')
		temperature = ss.get('temperature', 0.0)
		hyde = ss.get('use_hyde')
		hyde_prompt = ss.get('hyde_prompt')
		if ss.get('use_hyde_summary'):
			summary = ss['index']['summary']
			hyde_prompt += f" Context: {summary}\n\n"
		task = ss.get('task')
		max_frags = ss.get('max_frags',1)
		n_before = ss.get('n_frag_before',0)
		n_after  = ss.get('n_frag_after',0)
		index = ss.get('index',{})
		with st.spinner('preparing answer'):
			resp = query(question, index,
					task=task,
					temperature=temperature,
					hyde=hyde,
					hyde_prompt=hyde_prompt,
					max_frags=max_frags,
					limit=max_frags+2,
					n_before=n_before,
					n_after=n_after,
					model=ss['model'],
				)
		usage = resp.get('usage',{})
		usage['cnt'] = 1
		ss['debug']['query.resp'] = resp
		ss['debug']['resp.usage'] = usage
		ss['debug']['vector_query_time'] = resp['vector_query_time']
		
		q = question.strip()
		a = resp['text'].strip()
		ss['answer'] = a
		output_add(q,a)
		st.experimental_rerun() # to enable the feedback buttons

def b_clear():
	if st.button('clear output'):
		ss['output'] = ''

def b_reindex():
	# TODO: disabled
	if st.button('reindex'):
		index_pdf_file()

def b_reload():
  if st.button('reload prompts'):
      import importlib
      #importlib.reload(prompts)
      TASK = TASK

def b_save():
	db = ss.get('storage')
	index = ss.get('index')
	name = ss.get('filename')
	api_key = ss.get('api_key')
	disabled = not api_key or not db or not index or not name
	help = "The file will be stored for about 90 days. Available only when using your own API key."
	if st.button('save encrypted index in ask-my-pdf', disabled=disabled, help=help):
		with st.spinner('saving to ask-my-pdf'):
			db.put(name, index)

def b_delete():
	db = ss.get('storage')
	name = ss.get('selected_file')
	# TODO: confirm delete
	if st.button('delete from ask-my-pdf', disabled=not db or not name):
		with st.spinner('deleting from ask-my-pdf'):
			db.delete(name)
		#st.experimental_rerun()

def output_add(q,a):
	if 'output' not in ss: ss['output'] = ''
	q = q.replace('$',r'\$')
	a = a.replace('$',r'\$')
	new = f'#### {q}\n{a}\n\n'
	ss['output'] = new + ss['output']

# LAYOUT

with st.sidebar:
	ui_info()
	ui_spacer(2)
	with st.expander('advanced'):
		ui_show_debug()
		b_clear()
		ui_model()
		ui_fragments()
		ui_fix_text()
		ui_hyde()
		ui_hyde_summary()
		ui_temperature()
		b_reload()
		ui_task_template()
		ui_task()
		ui_hyde_prompt()


ui_api_key()
ui_pdf_file()
ui_question()
ui_hyde_answer()
b_ask()
ui_output()
ui_debug()
