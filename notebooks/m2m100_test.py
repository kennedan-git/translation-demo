# Databricks notebook source
pip install sentencepiece

# COMMAND ----------

from transformers import M2M100Tokenizer, M2M100ForConditionalGeneration

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M")



# COMMAND ----------

# MAGIC %md
# MAGIC Translate some random wikipedia text to portuguese.

# COMMAND ----------

text_to_translate = "The version of the useless machine that became famous in information theory (basically a box with a simple switch which, when turned ""on"", causes a hand or lever to appear from inside the box that switches the machine ""off"" before disappearing inside the box again[2]) appears to have been invented by MIT professor and artificial intelligence pioneer Marvin Minsky, while he was a graduate student at Bell Labs in 1952.[3] Minsky dubbed his invention the ""ultimate machine"", but that sense of the term did not catch on.[3] The device has also been called the ""Leave Me Alone Box"""
model_inputs = tokenizer(text_to_translate, return_tensors="pt")

# translate to French
gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("pt"))
print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))

# COMMAND ----------

import pandas as pd
test_df = pd.DataFrame({"id":[1,2,3], "content": ["Abraham Lincoln cut down a cherry tree", "Florida has nice beaches", "Elon Musk owns Tesla"],\
                        "src_lang": ["en", "en", "en"], "target_lang": ["pt", "ps", "es"]})

# COMMAND ----------

model_inputs = tokenizer(test_df.content.values.tolist(), return_tensors="pt", padding=True)

# translate to French
gen_tokens = model.generate(**model_inputs, forced_bos_token_id=tokenizer.get_lang_id("pt"))
print(tokenizer.batch_decode(gen_tokens, skip_special_tokens=True))

# COMMAND ----------

# MAGIC %md
# MAGIC Hindi and Chinese Example

# COMMAND ----------

hi_text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
chinese_text = "生活就像一盒巧克力。"
tokenizer.src_lang = "hi"
encoded_hi = tokenizer(hi_text, return_tensors="pt")
int_tkid = tokenizer.get_lang_id("en")
tkid = str(int_tkid)
generated_tokens = model.generate(**encoded_hi, forced_bos_token_id=int_tkid)
print(tkid)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# COMMAND ----------

# translate Chinese to English
tokenizer.src_lang = "zh"
encoded_zh = tokenizer(chinese_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# COMMAND ----------

#The Advocate chinese, a longer chinese example. 
chinese_text = "观音亭，恰在市街的中心，观音亭口又是这县城第一闹热的所在；就这个观音亭也成为小市集。由庙的三穿进入两廊去，两边排满了卖点心的担头，「咸甜饱巧」，各样皆备，中庭是恰好的讲古场；叹服孔明的，同情宋江的，赞扬黄天霸的，婉惜白玉堂的等等的人，常挤满在几条椅条上；大殿顶又被相命先生的棹仔把两边占据去，而且观音佛祖又是万家信奉的神，所以不论年节，是长年闹热的地方。\
后殿虽然也热闹，却与前面有些不同，来的多是有闲工夫的人，多属于有识阶级，也多是有些年岁的人，走厌了妓寮酒馆，来这清净的地方，饮着由四方施舍来的清茶，谈论那些和自己不相干的事情；而且四城门五福户的总理，有事情要相议，也总是在这所在，就是比现时的市衙更有权威的自治团体──所谓乡董局也设在这所在，所以这地方的闲谈，世人是认为重大的议论，这所在的批评，世间就看做是非的标准。\
　　但是来这所在的人，虽然是具有智能的阶级，却是无财力的居多，因为有财力的乡绅，自有他妻妾的待奉，不用来这所在的消耗他的闲岁月。因为这样关系，这所在的舆论，自然就脱离了富户人的支配，这些事情对于林先生的故事，也是真有影响。\
　　志舍自林先生走后，平添了无数烦恼，这烦恼虽不是林先生作弄出来的，但以前确是未曾有过。怎样一时百姓会不驯良起来，本来是交了钱，才去做风水，现在死人埋下去后还是不交钱，管山的虽然去阻挡，大家总是不听，甚至有时还受到殴打。像我们这地方，有几万人的城市，一日中死的是不少人，全都是扛到山顶去埋葬，这是志舍一个真大的财源，现在看看要失去了，他怎会甘心，就仗着钱神的能力，去要求官府的保护。\
　　不先不后，同这时候，林先生也向官府提出告诉去。告的是：志舍不应当占有全部山地做私产。他的状纸做得真好，一时被全城的百姓所传诵。大意是讲：「人是不能离开土地，离去土地人就不能生存，人生的幸福，全是出自土地的恩惠，土地尽属王的所有，人民皆是王的百姓，所以不论什么人，应该享有一份土地的权利，来做他个人开拓人生幸福的基础；现在志舍这人，没有一点理由，占有那样广阔的山野田地，任其荒芜墟废，使很多的人，失去生之幸福的基础，已是不该，况且对于不幸的死人，又徵取坟地的钱，再使穷苦的人弃尸沟渠，更为无理。所以官府须把他占有权剥夺起来，给个个百姓，皆有享用的机会，又可以尽地之利，是极应当的事，官府须秉王道的公平，替多数的百姓设法。」\
　　这张状纸会被这样多数的人所传诵，就因为这意见是大家所赞成的，不单止是城市里的人，就是村庄的做穑人，听着这事也都欢呼起来；多数的人──可以讲除起志舍一派以外，多在期待着这风声能成为事实，同时林先生也就为大家所爱戴了。\
　　本来百姓的愿望，不能就被官府所采纳，因为百姓有利益的事，不一定就是做官人的利益，像林先生所提起的告诉，虽然是为着无钱的百姓们的利益，又不和官府的利益相冲突，但是做官人完全得不到利益，做官的是不缺少五钱银买坟地的钱，甚不以林先生的告诉为是；一面志舍又在要求保护他的利益，究竟还是钱的能力大，所以官府把百姓们不遵向来的惯例，不纳志舍的钱，便讲是林先生煽动的，用那和谋反一样重大的罪名──扰乱安宁秩序的罪，加到林先生身上，把林先生拿去坐监。\
　　百姓们听到这消息，可就真正骚扰起来了，尤其是大多数无钱的人，更较激昂。\
　　「为着大家的事，把林先生拿去坐监，这是什么官府？」\
　　「□我们大家的俸禄，却专保护志舍一家，□钱官！」\
　　「打！打到志舍家里去！」"


# COMMAND ----------

tokenizer.src_lang = "zh"
encoded_zh = tokenizer(chinese_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_zh, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=False)

# COMMAND ----------

#Urdu example: Universal declaration of human rights preamble. 
urdu_text = "انسانی حقوق کا عالمی منشور\
اقوامِ متحدہ کی جنرل اسمبلی نے ۱۰؍ دسمبر ؁۱۹۴۸ء کو ”انسانی حقوق کا عالمی منشور“ منظور کر کے اس کا اعلانِ عام کیا۔ اگلے صفحات پر اس منشور کا مکمل متن درج ہے۔ اس تاریخی کارنامے کے بعد اسمبلی نے اپنے تمام ممبر ممالک پر زور دیا کہ وہ بھی اپنے اپنے ہاں اس کا اعلانِ عام کریں اور اس کی نشر و اشاعت میں حصہ لیں۔ مثلاً یہ کہ اسے نمایاں مقامات پر آویزاں کیا جائے۔ اور خاص طور پر اسکولوں اور تعلیمی اداروں میں اسے پڑھ کر سنایا جائے اور اس کی تفصیلات واضح کی جائیں، اور اس ضمن میں کسی ملک یا علاقے کی سیاسی حیثیت کے لحاظ سے کوئی امتیاز نہ برتا جائے۔"

tokenizer.src_lang = "ur"
encoded_ur = tokenizer(urdu_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_ur, forced_bos_token_id=tokenizer.get_lang_id("en"))
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

# COMMAND ----------


