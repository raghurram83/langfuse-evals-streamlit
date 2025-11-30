"""
Conversation → FAQ Extractor

Features:
- Upload transcript (.txt, .csv, .json) or paste raw text
- Normalization with speaker detection
- PII redaction (remove names, business names, phone numbers; keep cities/locations/products)
- LLM-based FAQ extraction (OpenAI gpt-4.1-mini by default)
- Side-by-side pipeline view (raw/cleaned vs normalized/FAQ)
- Confidence-colored FAQs and CSV export

Running locally: streamlit run app.py
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from openai import OpenAI, OpenAIError
import sys

# Ensure UTF-8 for I/O to avoid ascii codec issues.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass


# -----------------------------
# Requirements generator section
# -----------------------------
REQUIREMENTS = ["streamlit", "pandas", "openai"]


def sanitize_text(text: str) -> str:
    """Remove problematic Unicode separators that can break encoding."""
    return text.replace("\u2028", " ").replace("\u2029", " ").strip()


SAMPLE_TRANSCRIPTS = {
    "Agent Nisha - IVR pricing (en)": """[00:00:00] Customer: Hello.
[00:03:070] Agent: Yeah, hello.
[00:04:150] Agent: Welcome to My Operator, sir. Nisha this side. How may I help you?
[00:08:070] Customer: Uh, sorry your name?
[00:10:878] Agent: Uh, Nisha.
[00:12:968] Customer: Nisha. Okay. Uh, Nisha, my name is Junaid. I'm calling you from uh Sai Private Limited. Uh, we are looking for IVR services. Do you provide?
[00:22:908] Agent: Yes, sir. We do provide IVR services.
[00:26:058] Customer: Okay. We are a startup company. We are maximum three to four users and we are, we want these IVR services purely for our uh incoming calls or inbound calls. So, we have a specific requirement and maybe over a period of time we can scale it. What exactly what I'm looking is for uh, we have our official number listed on Google and uh other resource of channel like website and so on.
[00:55:128] Agent: Okay.
[00:55:518] Customer: And we need your DID number where we can forward those calls to your DID number and the users can get those calls accordingly.
[01:03:778] Agent: Got it.
[01:06:918] Agent: So, sir, this all can be possible with our uh plan. So basically in this plan one virtual number on which what we can do is in front for we can keep your company number and we can put the call forwarding to our virtual number. So, IVR will get played. Okay? And then later on it can be routed to your different users.
[01:31:118] Agent: Okay? IVR is customized, you can customize according to your needs. It's a very DIY process. You just need to write the prompt. Welcome to ABC company, like that.
[01:40:918] Customer: Yeah, I'm aware of that because I've been using uh, MQ, Tele CRM, Tele CMI in the past. But I was not sure whether you can customize and offer it. I know how IVR works. I have a clarity about it.
[01:52:438] Agent: So basically all this will be there in the idea.
[01:55:018] Customer: And also, uh, yeah, with which companies your DID number the you are registered with? Does Hotel? Exotel?
[02:05:048] Agent: With which company ID number is registered with?
[02:10:328] Customer: Yeah.
[02:11:158] Agent: Sir, basically, uh we just get our numbers from the Telco.
[02:17:348] Customer: Telco. Okay.
[02:18:678] Agent: Yeah. So, they provide us the virtual number and we just.
[02:22:218] Customer: So, because every company, every operator have, they need a different set of documents. For example, someone needs uh founders Aadhaar card, PAN card, GST number. Some needs one uh DNBC uh certificate signature something like such. So that is the reason I'm asking you, these documents are there any specific requirements that you guys require?
[02:44:738] Agent: So basically sir, we are connected with four servers, Tata, VI, Jio, Acel. And the Telco provides us the number or the virtual numbers and then we sign it to you.
[02:58:158] Customer: Okay. And the virtual number uh it is a new number, right? It's not a used number or anything?
[03:03:228] Agent: No, no, no. It's a new number, it's a fresh DID which we we will be providing you. And yeah, that's it. It is a new DID and it would be starting from 080 series or 022 series.
[03:17:158] Customer: Okay. Okay. So what are your plans? What is the pricing?
[03:21:638] Agent: Uh sir, I have just one question. Do you want CRM integration as well?
[03:26:938] Customer: We have our Zoho Begin CRM, yes. So we need the CRM integration.
[03:33:418] Agent: Okay, with Zoho Begin.
[03:35:938] Customer: Yeah, we have Zoho Begin. We are using Zoho Begin.
[03:39:878] Agent: Okay. So, basically, we have two plans for you. One is the sedan plan and one is the SUV plan. In sedan plan, there is no CRM integration, but in the SUV, we can provide you the CRM integration. That plan starts from five users in which it's just a click uh just one click uh integration in which you can receive unlimited incoming calls, you can do unlimited outgoing calls as well. There is no additional or hidden charge on that. Okay? With this, a multi-level IVR will be there that means the different layers will be there, one for the language, one for the departments and all. Okay? So if you I think.
[04:20:808] Customer: That is not required for me. I am going, I was just checking your uh uh pricing. I see compact plan which is 2,500 rupees for three users.
[04:31:678] Agent: That just, that is just for the WhatsApp business API, the compact plan.
[04:37:378] Customer: What you have mentioned as one phone number.
[04:39:698] Agent: Yeah. So, for the WhatsApp, we provide a one virtual number and WhatsApp as activated on that number.
[04:48:718] Customer: So that, so the compact plan does not have an IVR facility?
[04:52:168] Agent: No, it doesn't have any IVR facility. You can see the Sedan and the SUV plan.
[04:58:638] Customer: Okay.
[04:59:618] Agent: So, in Sedan, there is IVR facility, but there is no integration. But in SUV plan, we have IVR as well as the integration.
[05:09:868] Customer: Okay, leave about integration. Maybe later step we can do it. Uh when you.
[05:14:388] Agent: Then you can go for the Sedan plan.
[05:16:168] Customer: Yeah.
[05:16:718] Agent: So, it's of 60,000 rupees uh in which uh IVR will be there. WhatsApp business API is also an add-on in this package. Call recordings will get all the call recordings, you can download those call recordings. Contacting feature is there, from which location you are receiving the calls. You can uh just see the break time management as well, at what time you are receiving most of the incoming calls. So, there would be a graph there over there on our platform. And you can see it and when you are receiving most of the calls at that time, you can keep your user free so no calls are getting missed.
[05:52:358] Customer: But I don't have that many users. So I don't think so this plan will suit me. As I mentioned to you, my requirement, we have three to four users.
[06:00:268] Agent: Yeah, I got it. But in check plan, you can see there are four pairs of channel lines. This is for 10 users, but only four. As you can see in the Sedan plan, there are only four pairs of channel lines written over there on the platform. You can see there. That means you can add 10 users but at that time only four calls will be connected. You have three to four users, it can suit you. It would be 60,000 rupees only.
[06:28:988] Customer: For whole year.
[06:30:868] Customer: The pricing is very high. If I compare to MQ, if I compare to Tele CMI and there is one more, there is one more, the pricing is very high.
[06:38:628] Agent: What pricing were they giving you for three to four users for whole year?
[06:42:858] Customer: For two users, they had given me for four to five users, the pricing uh hold on. Let me check. Yeah. It was coming for approximately uh 39,000 to 41,000 including GST.
[07:03:318] Agent: No, no, no. My purpose, yeah. Just give me one second. I just uh see is it possible or not.
[07:14:738] Customer: Okay, sure.
[07:17:518] Agent: Just give me one second.
[07:19:938] Customer: Okay.
[07:23:438] Agent: Sir, I can split you for three users, the only incoming plan in which IVR will be here in 45,000 rupees excluding GST.
[07:35:868] Customer: Next to way high.
[07:36:918] Agent: And this is for three users only.
[07:41:408] Customer: I'm getting for 39 to 41,000 rupees for four users. Uh, why would I go for the three users with the same, I mean for the higher price because excluding GST.
[07:50:528] Agent: There are, there are many companies which are providing the same features. We are not the only company but when it comes to resources, as I told you earlier, we are running on four servers. If one server is low, we will switch to there is one AI that will switch to another server. So we have four servers that is Tata, VI, Jio, Airtel. Okay? Whenever there is any uh issue in the server, we just switch you within seconds. Second thing, our, we are, we provide you written guarantee of uptime guarantee of 99.9%. Theek hai? Iske sath, we use the one of the best uh data security uh app that is uh AWS, Amazon web server. And that is best for the data security and privacy. Once you log in to our panel, we also don't have access to your panel. You need to recreate your password, then whenever you log in to our panel, one OTP is generated on the user phone number which you have written. And then you need to put that OTP, tabhi aap login kar sakte hai panel pe. So we have provided you all those things which are necessary. IVR is just a very DIY process, easy to use.
[08:58:248] Customer: See, I, I mentioned to you in the beginning of the call, we are a startup. So definitely we will scale quarter to quarter, we are not going to be on the same thing, that is for sure.
[09:10:508] Agent: I can provide you sir Sedan plan. Yeah. I can provide you the Sedan plan in 51,000 rupees and you can add up to 10 users but there would be four pairs of channel lines. If you want, I can provide you this offer. I can provide you up to 10% discount.
[09:30:178] Customer: Your Sedan, I mean your SUV plan and um Sedan plan, I mean can can it be customized like the the bare minimum what you can give me is for four users, okay? And we have to make one short payment?
[09:44:938] Agent: Yeah, you have to pay uh like it's a prepaid plan.
[09:50:088] Customer: No, I mean you don't you have monthly, quarterly?
[09:52:998] Agent: No, no, no, no. So, if you are going for half yearly sedan, it would be 6,000 rupees per month. So, it would be around 30, the cost keeps on increasing if you are uh you know, decreasing the validity validity.
[10:10:048] Customer: But again, it's uh cost effective for us. Right now, my requirement is very simple. It's not customized and complicated. I need an IVR where I can get those calls, those call recordings, those miss calls. Whatever the data is there in the system so that I can sync it to my Zoho Begin. So I have a track of it, so I would I would understand how many calls I'm getting on a day-to-day basis, how many leads has been successful or unsuccessful. That is the purpose of this entire uh IVR structure what I'm looking for. Right now I'm not tracking WhatsApp click tracking uh yeah.
[10:54:198] Agent: As I told you for six months it would be 36,000 rupees and for whole year I can provide you the up to 10% discount and it would be in 51,000 rupees for whole year.
[11:07:328] Customer: For how many users?
[11:09:228] Agent: For uh, if you want, you can use the four users. Later on if you want to add on more, up to 10 users would be there. There is no customize into this plan. I I've given you the discount of 10%.
[11:24:008] Customer: You're saying including GST.
[11:30:748] Agent: What sir?
[11:31:658] Customer: Hello.
[11:32:898] Agent: Yes, sir.
[11:33:438] Customer: Are you saying 51,000 including GST?
[11:36:768] Agent: No, 51,000 excluding GST.
[11:40:248] Customer: Just before now you said 45,000 plus GST, which was that plan that you mentioned?
[11:46:278] Agent: 45,000 plus, uh, just second. That was only incoming plan for three users.
[11:57:338] Customer: Aha.
[11:57:848] Agent: So that is 45,000 rupees for three users. Only incoming would be there. IVR would be there. Theek hai? And that would be of 45,000 plus 18% GST you you have to pay.
[12:14:048] Agent: So there are three plans which I have gave you. Only incoming plan that is of 45,000 rupees plus 18% GST for three users. Another plan is the sedan plan. It is of uh 60,000 rupees but I have given you the discounted price that is 51,000 rupees plus 18% GST. And the last plan that is sedan half yearly, that is of 36,000 rupees plus 18% GST for six months only.
[12:42:918] Customer: I don't think so this will serve our purpose because taking 10 users, paying 51,000 plus GST is not worth for us and the plan which you suggested me which is 45,000 per year for three users again it is very expensive because.
[12:58:658] Agent: Sir you already you know, paying 39 to 40,000 rupees for three users. As you told me. You're already paying 39 to 41,000 for three users.
[13:08:148] Customer: That includes the that includes the API integration along with the CRM. It's just not the IVR.
[13:16:798] Agent: In only incoming plan that is of 45,000 rupees I can provide you the CRM integration. Okay.
[13:25:298] Customer: Not worth, not worth. Anyways, thanks. If I think uh you know, something is possible then I'll reach out to you guys.
[13:33:148] Agent: Thank you. Okay, sure. Thank you so much.
[13:35:888] Customer: Bye.""",
    "Agent - WhatsApp order API (hi/en mix)": """[00:02:16] Customer: Hello.
[00:03:68] Customer: Hello.
[00:04:99] Agent: Welcome to My Operator sir. How may I help you?
[00:08:485] Customer: Sir humein na ek API ki requirement thi WhatsApp ki.
[00:11:935] Agent: Theek hai.
[00:13:815] Agent: Toh, aapka business kis cheez ka hoga?
[00:17:15] Customer: Hamara na, ek client hai mere. Unko required hai, unka, uhm, atta chakki ka business hai, jisme website se atta bechte hai.
[00:26:405] Customer: Theek hai?
[00:27:145] Agent: Okay, okay, okay.
[00:28:355] Agent: But mai bata deta hu mujhe requirement kya hai.
[00:31:135] Agent: Theek hai.
[00:31:405] Customer: Unhe chahiye ki jaise hi koi WhatsApp se, uh, order purchase ho.
[00:36:115] Agent: Theek hai.
[00:38:15] Customer: Sir like.
[00:38:525] Customer: Hume kuch aisa chahiye ki WhatsApp pe order purchase ho aur wo humare system ke saath integrate ho jaaye.
[00:45:625] Agent: Acha, system kis cheez ka?
[00:46:895] Customer: jaise hi purchasing ho,
[00:48:985] Agent: Humm.
[00:49:155] Customer: like jaise hi WhatsApp se purchasing ho, to ek API mujhe data returning kar de, ki yaha se is user ne yeh, yeh cheez purchase kari hai.
[00:57:385] Customer: Kya yeh possible hai?
[00:59:615] Agent: Theek hai.
[01:00:875] Agent: Uhm. Matlab basically aur aap isse integrate kis se karwayenge? Kis software se apne?
[01:05:465] Customer: Yeh humara custom software hai like mera Wordpress pe bana rakha hai, coordinator pe hai. PHP coordinator me hai yeh.
[01:13:305] Agent: Acha custom software?
[01:15:395] Agent: Naam kya bataya aapne?
[01:16:355] Customer: Custom software hai, custom.
[01:17:755] Customer: coordinator
[01:19:625] Agent: haa.
[01:20:25] Agent: Coding, aage?
[01:22:315] Customer: coordinator.
[01:24:255] Agent: coordinator. Theek hai. Toh aap basically yeh chahte ho ki WhatsApp
[01:27:265] Customer: framework hai.
[01:28:445] Agent: Theek hai.
[01:28:835] Agent: basically aap yeh chahte ho ki jab bhi kabhi koi order place ho toh usme kya hona chahiye ki, uh, jaise WhatsApp pe woh order place ho gya. Toh uska saari details wagerah jo bhi cheez rahegi, wo aapke, jo apna portal hai aapka, jo apna software hai uske upar aa jaani chahiye, right?
[01:45:845] Customer: Haanji, suppose ise dekhiye. Jaise maan lijiye maine ek product ka link bheja.
[01:49:735] Agent: Theek hai?
[01:50:35] Customer: Client ko WhatsApp pe.
[01:51:745] Agent: Acha.
[01:53:775] Customer: Theek hai? Jab woh link gaya, client ke pass ek form khul jayega.
[01:57:465] Customer: Usme basic details hongi, address hoga, yeh sab hoga. Jaise hi woh form fill karega WhatsApp pe, toh jo woh fill karega waise hi mere pass ek API mujhe returning data degi jisme main us data ko apne system me save kar lunga ki order confirmation le lunga main client ki.
[02:18:965] Agent: Hello.
[02:19:355] Customer: Hello.
[02:20:315] Agent: Yes sir?
[02:21:405] Customer: Haanji.
[02:22:255] Agent: Toh, sir ek baar fir se aap repeat karenege, yeh, yeh part? Actually audible nahi thi voice proper.
[02:28:815] Customer: Main yeh chahta hu,
[02:30:605] Customer: Maan lijiye meri site se ek link generate hoga.
[02:33:485] Agent: Theek hai?
[02:33:635] Customer: jo client ke pass jayega.
[02:35:145] Agent: Theek hai?
[02:35:715] Customer: Chahe wo main manual bheju, chahe wo main apne system me lagau use. Jaise bhi matlab ki maan lijiye ek link generate kar liya maine. Wo link client ke WhatsApp pe gaya.
[02:45:265] Customer: Jaise hi client uspe click karta hai, ya to wahan click ho jaaye ya fir jaise hi link jata hai toh uske pass ek form khul jayega.
[02:53:715] Agent: Humm.
[02:54:995] Customer: Form khula, form fill hua. Jaise hi form fill hota hai,
[02:58:805] Customer: jo bhi uske andar data hota hai form fill hone ke baad submit karta hai woh, wo data ka returning mujhe chahiye.
[03:04:815] Customer: ki yeh product tha.
[03:06:405] Customer: Uske upar usne yeh naam daala, apna yeh address daala, sab kuch daalne ke baad wo mere pass aa jaye aur main use apne order wale section me add kar lunga.
[03:14:145] Agent: Theek hai. Theek hai.
[03:16:795] Customer: Haanji.
[03:17:155] Agent: Theek hai.
[03:18:295] Customer: Theek hai.
[03:18:795] Agent: Aur kuch is, iske alawa bhi kuch aur requirements hai?
[03:22:425] Customer: Bas abhi mujhe yahi chahiye.
[03:25:5] Agent: Theek hai. Toh yahan par ek cheez mujhe samjhna hai ki jo link hogi, theek hai, yeh link kahi na kahi hum hi provide kar rahe hai aur link ke andar jo content hoga wo bhi kahi na kahi hum hi decide kar rahe hai ki kya content hona chahiye, right?
[03:37:125] Customer: Aap mujhe decide karke bata dijiye, kya ho sakta hai usme. Kya kya aapko required hai? Maan lijiye product ki ID chahiye, product ka name chahiye aapko.
[03:44:845] Agent: Humm.
[03:44:985] Customer: Theek hai?
[03:46:115] Customer: Jo jo aapko data required hoga wo main us link me aapko de dunga.
[03:49:875] Agent: Theek hai.
[03:51:305] Customer: Theek hai?
[03:51:775] Customer: Us link me jaise hi wo data aapke, maan lijiye wo data client ke pass gaya.
[03:57:125] Customer: Jaate hi saari ek form khul jayega ki order now ka button aa jayega WhatsApp pe. Order now kiya. Order now me WhatsApp ke andar ek form khul jayega ki kitni quantity chahiye aapko? kya aapka naam hai? kya aapka address hai? Wo sab usne fill kiya basic details aur usko submit kiya.
[04:11:825] Agent: Humm.
[04:12:355] Customer: Toh wahan se jo data hai wo meri ek API ko hit kar dega.
[04:15:355] Customer: Aur us API me jo aap mujhe returning data denge, usko main apne pass save kar lunga to kind of mere liye ek order generate ho jayega.
[04:22:155] Agent: Toh kind of main yeh samajh paa raha hu ki yeh form hoga matlab basically WhatsApp forms jo hote hai.
[04:27:775] Agent: ki jahan par basically jitne bhi information hoti hai wo saari wahan par hogi, right?
[04:31:865] Customer: Jee, jee, jee, jee, jee.
[04:33:655] Agent: Theek hai, theek hai, theek hai, theek hai.
[04:35:985] Agent: Theek hai karwa dete hai sir. Isme hum
[04:37:375] Customer: Wahan se, wahan se ek form me mereko ek returning API me mujhe saara data receive ho jayega.
[04:43:245] Agent: Theek hai, matlab integration karna hoga humko uske liye.
[04:46:665] Agent: Aur theek hai toh kab aap available hai mujhe...
[04:48:475] Customer: Basically aap sirf mujhe ek JSON format me de dijiyega.
[04:51:565] Agent: API haa.
[04:52:135] Customer: Ek JSON format me callback de dijiyega, usko fir main apne isme integrate kar lunga.
[04:56:885] Agent: Theek hai, theek hai got it.
[04:58:655] Agent: Aah, ek kaam karte hai sir. Ek baar na agar aap available ho shaam me,
[05:04:135] Agent: toh ek baar main isse live ek baar aapko main dikhane ki koshish karta hu ki cheezein kaise hongi. Ek baar wahan pe discuss kar lete hai aur fir matlab agar aapko lagta hai toh we can proceed fir.
[05:14:485] Customer: Theek hai. Uski thoda mujhe costing bhi bata dijiyega kya rehta hai aapka?
[05:18:855] Agent: Aah, costing sir, actually na yeh jo feature hai particularly form,
[05:23:445] Agent: toh iski hi additional cost aati hai. Toh woh additional cost mujhe puchna padega kyunki hum isse customize karenge aapke liye.
[05:30:175] Agent: Waise lumpsum jo aa raha hota hai woh aap yeh samjhiye 12 se 15 hazar rupaye ki cost aati hai ek form ki sirf.
[05:35:55] Agent: Theek hai? But ek baar main confirm kar leta hu ki poora solution jo humara hoga, woh solution ka poora cost kya aayega, actual cost kitna aayega?
[05:42:385] Customer: Actual cost kitna aayega aur aap kaise lete hai? Ek form khulne ke upar payment hota hai ki like, one time payment hota hai? Kaise hota hai? Wo thoda sa mujhe saara ek baar detail de dijiyega kyunki mujhe yeh detail client ko deni hai.
[05:52:655] Agent: Done done done aapko proposal.
[05:53:775] Agent: Acha acha aapko clients ko share karna hai. Theek hai.
[05:56:45] Customer: Haan haan haan. Actually mera software development ka hi kaam hai.
[05:58:935] Customer: Toh hume requirements rehti hai APIs ki.
[06:01:835] Agent: Haan haan haan. Yeh ek baar aap na poora format bataiyega apna kitna hai, kaise hai?
[06:05:45] Customer: Main ek baar
[06:05:955] Agent: Done done done.
[06:06:915] Agent: Main ek baar product team ko isme involve kar leta hu. Theek hai?
[06:09:475] Agent: Maybe agar unke end se kuch questions honge na, toh main fir se aapko call kar lunga. Isi number pe call kar lunga. Theek hai?
[06:15:35] Agent: Taaki ek actual cost aap bhi aage matlab clarity ke saath share kar paaye ki haan is, yeh humara project hai aur yeh itni cost aa rahi hai aur is tarike se aayegi.
[06:21:955] Customer: Haanji.
[06:22:925] Agent: Hai na?
[06:22:965] Agent: Toh main ek baar product team se confirm kar leta hu. Baaki agar kuch required hota hai toh main aapko fir se call kar lunga isi number se.
[06:28:445] Agent: Aur baaki main proposal to aapko share karwa dunga.
[06:30:555] Customer: Jo abhi aapko jo jo karwana hai.
[06:31:735] Agent: Haanji. Abhi jo humara jo data hai na,
[06:35:105] Agent: abhi jo maan lijiye fields humne isme rakhni hai.
[06:37:115] Agent: Ek baar aap mujhe thoda sa bata dijiye.
[06:39:795] Agent: Fir main aapko final fields bataunga client ke saath connect karke.
[06:43:845] Agent: Ki wahan par kya kya chahiye.
[06:46:175] Customer: Hmm hmm.
[06:47:485] Agent: Haan haan.
[06:47:895] Agent: Main abhi aapko basic proposal to bilkul pahunchwa de raha hu. But mere end pe na mujhe bhi ek clarity chahiye ek amount commit karne se pehle. Main ek baar confirm kar leta hu fir aapko main proposal ke form me share karwa dunga. Saari jisme saari cheezon ki clarity hogi aapko.
[06:59:755] Customer: Theek hai ji. Theek hai.
[07:01:425] Agent: Theek hai?
[07:02:185] Customer: Okay sir.
[07:02:715] Agent: Okay ji.
[07:03:975] Customer: Theek hai ji.
[07:04:475] Agent: Okay ji.""",
    "Agent - WhatsApp banking API (hi/en mix)": """[00:00:00] Customer: Hello.
[00:00:02] Customer: Hello. Uh
[00:00:04] Agent: Yeah?
[00:00:04] Customer: Hello.
[00:00:05] Agent: Yeah, welcome to my operator, sir.
[00:00:10] Customer: Hello. My, My crystal, my crystal. Han. Sir ye main ye janna chah raha hu main actually bank se baat kar raha hu. Ye aap whatsapp banking service provide karte ho?
[00:00:21] Agent: Right, right unke liye. Sir, hum like hum cam, hum jo platform hota hai basically jaha se aap operate kar sakte ho, hum vo provide karte hai.
[00:00:29] Customer: Acha.
[00:00:30] Agent: Haanji.
[00:00:31] Customer: To aap matlab API service API provide karte ho? WhatsApp?
[00:00:34] Agent: Han. Haanji, WhatsApp API provide karte hai.
[00:00:37] Customer: To aapka kisi bank ke sath to hai tie up?
[00:00:41] Agent: Kisike sath?
[00:00:41] Customer: Main bank banking bank ko aap dete ho service? Kisi bank ko ya financial institution ko?
[00:00:48] Agent: Uh sir matlab kis cheez ki financial hai, aap thoda sa batayenge? Like uh thoda matlab uh SEBI se related hai aapka financial advisor?
[00:00:54] Customer: No, no, no hum, hum, haan. Dekho, hum, haan hum, humari banking le rahi hai, whatsapp banking ki service.
[01:00] Customer: Hum whatsapp banking ke liye, aapki whatsapp bank ki ki API use karni hai. Jiske through hum usme us service ko hum as a marketing tool use kar sakte hai, ya bank, bank ki jo matlab koi balance enquiry, banking ki jo service hai, account service customer ki jo hai usme humne hume use karni hai. To main isliye puch kabhi aapse puch raha hu ki aap bank ko service dete ho?
[01:22] Agent: Sir, main, main bhi aapse sir like main vo cheez aapki samajh gaya. Hum dete hai banks ko services, lekin vo depend karta hai ki bank ke end se kaisi services use ho rahi hai.
[01:30] Agent: Agar to sir jaise ab baat karu, agar aap mutual funds ya like financial advisories ya cold messages bhejenge to usme nahi hota.
[01:38] Customer: Na, na, na. Customer ko customer ko provide karni hai service, theek hai. Customer ko customer ko service provide karni hai. Jaise ki vo apna balance check kar sake, koi bhi jo is tarah ki banking ki service hai, account statement hai, is tarah ki service puch raha hu.
[01:52] Agent: Accha, accha. To, haan, is, is cheez me to sir aapka possible hai. Very much possible hai aapka. Bas ye cheez aapko dhayan rakhna hoga ki aapka cold messages nahi hone chahiye. Jaise ki aap randomly kisi ko message nahi kar rahe ya to data hona chahiye ya lead generation form se aapke pas customers ka data hona chahiye.
[02:07] Customer: Vo, vo, vo, to, haan, haan. Vo to hum aapko provide karenge na. Vo to registration hogi na customer ki jo authorized hum hoga customer bank ka customer hoga to vohi to vo register kar payega na. To tabhi to vo service use kar payega na?
[02:20] Agent: Haanji, haanji.
[02:23] Customer: Nai
[02:23] Agent: Abhi tak to sir isme yahi rehta hai. Haan. Isme yahi rehta hai. Main aapko thoda clearly samjhata hu. Aap use kar sakte ho, koi dikkat nahi hai.
[02:30] Agent: But ye hota hai ki jaise ki koi matlab false commitment nahi honi chahiye ya like that nahi hota ki theek hai advices jo hai vo galat nahi honi chahiye. In that case to vo cheez rehti hai iske andar.
[02:41] Customer: Theek hai. Suno, suno, suno. Humne thode dino me na RSP publish karne wale hain. Agar aapke regarding hogi vo RSP aap dekh lena.
[02:49] Customer: To aap mere ko bata dena. Theek hai?
[02:51] Agent: Theek hai, theek hai, theek hai. Aap isi number pe call kar sakte ho dobara to hum dobara connect kar sakte hai.
[02:56] Customer: Aap ek baar dekh lijiye.
[02:57] Agent: Haan, main connect karunga lekin haan haan.
[02:59] Customer: Humaari bank me humari kuch requirement hai to humari, humari, humara government organisation hai.
[03:05] Customer: To jiski vajah se hume RSP publish karni padti hai. Directly hum nahi ja sakte kisi bhi vendor ke paas.
[03:11] Customer: To us RSP ke against aapko aap check kar lena ki humari kya requirement hai ya clauses hai vo aap check kar lena. Agar aapki company ko suitable hoga to aap dekhna apply kar dena.
[03:23] Agent: Theek hai, theek hai. Theek hai sir, theek hai.
[03:25] Customer: Theek hai?
[03:26] Agent: Haanji.
[03:27] Agent: Okay sir. Okay.
[03:30] Agent: Okay.""",
    "Agent Rahul - renewal (en)": """[00:00:00] Agent: Welcome to My Operator. My name is Rahul. How may I assist you?
[00:00:04] Customer: My name is Ashish.
[00:00:06] Customer: I am speaking from Calcutta.
[00:00:08] Customer: We have a toll-free number running
[00:00:11] Customer: with My Operator.
[00:00:12] Customer: And I want to talk to my account manager actually.
[00:00:16] Agent: Understand, sir.
[00:00:17] Agent: Let me first check with your account, please be online.
[00:00:21] Customer: Yes.
[00:00:25] Agent: In the meanwhile, can you please confirm the concern, sir? so that I can coordinate accordingly.
[00:00:30] Customer: Sorry?
[00:00:31] Agent: Can you please confirm with the concern, sir? So that I can coordinate accordingly.
[00:00:35] Customer: I want to I want to talk about the renewal plan.
[00:00:39] Agent: Renewal plan, I got you, sir.
[00:00:43] Agent: Yeah, I got your detail.
[00:00:49] Customer: Yes.
[00:00:58] Agent: Okay, you are connected with Mr. Ankur.
[01:00] Agent: Just be online, sir.
[01:02] Agent: Let me try to schedule your call for
[01:05] Agent: call transfer, just allow me a moment.
[01:07] Customer: Please, please.
[01:29] Agent: Okay.
[01:30] Agent: So I'm so sorry sir to put your call on wait. I'm unable to transfer the call. So what exactly I am doing? I'm directly sharing the details
[01:38] Agent: with Mr. Ankur to coordinate with you.
[01:41] Agent: But before that, I'm just want to confirm
[01:44] Agent: this will be the same connecting number to get back to you? 9650897009
[01:51] Customer: Yes, yes, on the same number.
[01:53] Agent: Alright sir, thank you so much to confirm that.
[01:58] Customer: Please ask Ankur to connect me as soon as possible.
[02:03] Agent: Sure, sure to be very frank, sir.
[02:06] Customer: Okay.
[02:07] Agent: Is there anything else I can assist you with with My Operator, sir?
[02:10] Customer: No. If possible, can you give me the number of Ankur?
[02:17] Agent: Yeah, just a moment, sir.
[02:42] Agent: Yeah, please note down sir.
[02:44] Customer: Hmm.
[02:46] Agent: It's 8069
[02:49] Customer: Yes.
[02:50] Agent: 158118
[02:54] Customer: 158
[02:56] Agent: 118
[02:59] Customer: 118. I am repeating it 8069158118
[03:07] Agent: 158118, right sir, you can also drop him a WhatsApp over WhatsApp.
[03:16] Customer: Okay.
[03:16] Agent: If he, if he may not respond you may drop message.
[03:21] Agent: However, I've already shared the detail to get connect with you sir.
[03:24] Customer: Okay okay. Thank you.
[03:26] Agent: Thanks you too sir and have a good day ahead.
[03:28] Customer: And I I'll I'll wait for some time and I'll dial myself.
[03:33] Agent: Uh, you can uh it's better like I am requesting you to wait else, I'll, uh, you may also connect via that particular number, sir.
[03:40] Customer: I will wait.
[03:42] Customer: If I don't get a call, then I'll connect myself.
[03:46] Agent: Sure sir. Sure, sir. Right.
[03:49] Agent: Okay, all right, sir. Have a good time ahead sir."""
}

# -----------------------------
# Parsing & normalization
# -----------------------------
def parse_transcript(content: bytes, filename: str) -> List[str]:
    """Convert uploaded content into a list of message lines."""
    name = filename.lower()
    text_lines: List[str] = []
    def _clean(line: str) -> str:
        return sanitize_text(line)
    if name.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")
        text_lines = [_clean(line) for line in text.splitlines() if _clean(line)]
    elif name.endswith(".csv"):
        df = pd.read_csv(io.BytesIO(content))
        # heuristics: use first text-like column
        text_col = None
        for col in df.columns:
            if df[col].dtype == object:
                text_col = col
                break
        if text_col:
            text_lines = [_clean(str(x)) for x in df[text_col].dropna().tolist() if _clean(str(x))]
    elif name.endswith(".json"):
        data = json.loads(content.decode("utf-8", errors="ignore"))
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # pick first string value
                    for val in item.values():
                        if isinstance(val, str):
                            cleaned = _clean(val)
                            if cleaned:
                                text_lines.append(cleaned)
                            break
                elif isinstance(item, str):
                    cleaned = _clean(item)
                    if cleaned:
                        text_lines.append(cleaned)
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, str):
                    cleaned = _clean(val)
                    if cleaned:
                        text_lines.append(cleaned)
    else:
        # fallback treat as text
        text = content.decode("utf-8", errors="ignore")
        text_lines = [_clean(line) for line in text.splitlines() if _clean(line)]
    return text_lines


def normalize_lines(lines: List[str]) -> pd.DataFrame:
    """Detect speaker labels and normalize to dataframe."""
    records = []
    for line in lines:
        line = sanitize_text(line)
        speaker = "Unknown"
        raw_text = line
        if line.lower().startswith("agent:"):
            speaker = "Agent"
            raw_text = line.split(":", 1)[1].strip()
        elif line.lower().startswith("customer:") or line.lower().startswith("user:"):
            speaker = "Customer"
            raw_text = line.split(":", 1)[1].strip()
        records.append({"speaker": speaker, "raw_text": raw_text})
    return pd.DataFrame(records)


# -----------------------------
# PII Redaction
# -----------------------------
PHONE_REGEX = re.compile(r"(?:\+?\d[\d\-\s]{6,}\d)")
BUSINESS_REGEX = re.compile(r"\b(?:inc|llc|ltd|corp|company|co\.|store)\b", re.IGNORECASE)
NAME_REGEX = re.compile(r"\b(Mr\.?|Ms\.?|Mrs\.?|Dr\.?|Prof\.?)\s+[A-Z][a-z]+|\b[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}\b")


def redact_text(text: str) -> str:
    """Heuristic redaction: remove names, business names, phone numbers; keep locations/products."""
    redacted = PHONE_REGEX.sub("[REDACTED_PHONE]", text)
    redacted = BUSINESS_REGEX.sub("[REDACTED_BUSINESS]", redacted)
    redacted = NAME_REGEX.sub("[REDACTED_NAME]", redacted)
    return redacted


def redact_pii(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cleaned_text"] = df["raw_text"].apply(redact_text)
    return df


# -----------------------------
# LLM FAQ extraction
# -----------------------------
FAQ_SYSTEM_PROMPT = """Extract possible FAQs from this conversation.

STRICT RULES:
- Only create FAQs that reflect CUSTOMER questions/intents (lines starting with 'Customer:').
- Ignore questions asked by the agent (e.g., "Do you have any more queries?", "How did you hear about the product?").
- Use Agent responses ONLY to supply answers; if the agent did not provide a useful answer, skip that FAQ.
- Do not invent or rephrase agent prompts as FAQs.

Identify question intent even when no '?' is present.
Respond ONLY with JSON (no prose, no markdown).
Return either a list or an object with key "faqs" containing the list.
Each FAQ object fields:
- question (string)
- answer (string)
- confidence (0-1)
- question_confidence (0-1)
- answer_confidence (0-1)
- relevance_score (0-1)
- completeness_score (0-1)
- redundancy_score (0-1)
- pii_removed (boolean)
"""


def extract_faq_via_llm(cleaned_transcript: str, model: str, api_key: str) -> List[Dict[str, Any]]:
    """Call OpenAI and robustly parse JSON FAQs."""
    if not api_key:
        raise ValueError("OpenAI API key is required for FAQ extraction.")
    client = OpenAI(api_key=api_key)
    messages = [
        {"role": "system", "content": FAQ_SYSTEM_PROMPT},
        {"role": "user", "content": cleaned_transcript},
    ]
    try:
        # GPT-5 models can be strict about params; send minimal payload.
        if model.startswith("gpt-5"):
            resp = client.chat.completions.create(model=model, messages=messages)
        else:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.2,
                max_tokens=800,
            )
        content = sanitize_text(resp.choices[0].message.content)
    except OpenAIError as exc:
        raise RuntimeError(f"OpenAI API error: {exc}") from exc

    faqs = _parse_json_loose(content)
    return faqs, content


def _strip_code_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        # remove first fence
        t = t.lstrip("`")
        # drop possible language hint
        parts = t.split("\n", 1)
        if len(parts) == 2:
            t = parts[1]
    if t.endswith("```"):
        t = t[: -3]
    return t.strip()


def _parse_json_loose(text: str) -> List[Dict[str, Any]]:
    """Try strict JSON first; then fallback to bracketed content extraction."""
    text = sanitize_text(text)
    text = _strip_code_fences(text)
    # Direct parse
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("faqs", "data", "items", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key]
    except Exception:
        pass
    # find first [...] block
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        snippet = text[start : end + 1]
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
        except Exception:
            return []
    return []


def color_by_confidence(conf: float) -> str:
    if conf >= 0.75:
        return "green"
    if conf >= 0.5:
        return "orange"
    return "red"


# -----------------------------
# FAQ Scoring
# -----------------------------
class FAQScorer:
    """Implements weighted scoring and decision logic for FAQs."""

    def score_faq(self, faq_dict: Dict[str, Any]) -> Dict[str, Any]:
        q_conf = float(faq_dict.get("question_confidence", 0) or 0)
        a_conf = float(faq_dict.get("answer_confidence", 0) or 0)
        rel = float(faq_dict.get("relevance_score", 0) or 0)
        comp = float(faq_dict.get("completeness_score", 0) or 0)
        red = float(faq_dict.get("redundancy_score", 0) or 0)
        pii_removed = bool(faq_dict.get("pii_removed", False))

        if not pii_removed:
            return {
                "overall_faq_score": 0.0,
                "decision": "REJECT",
                "debug_breakdown": {
                    "semantic_alignment": rel,
                    "answer_depth": a_conf,
                    "clarity": q_conf,
                    "pii_status": "FAILED",
                    "duplicate_penalty": red,
                    "final_weighted_score": 0.0,
                },
            }

        # Base weighted score
        score = (
            q_conf * 0.30
            + a_conf * 0.30
            + rel * 0.20
            + comp * 0.15
        )

        # Redundancy penalties
        penalty_applied = 1.0
        if red > 0.85:
            penalty_applied = 0.70
            score *= 0.70
        elif 0.60 <= red <= 0.85:
            penalty_applied = 0.85
            score *= 0.85

        # Clamp 0-1
        score = max(0.0, min(score, 1.0))

        if score >= 0.75:
            decision = "ACCEPT"
        elif score >= 0.55:
            decision = "REVIEW"
        else:
            decision = "REJECT"

        return {
            "overall_faq_score": score,
            "decision": decision,
            "debug_breakdown": {
                "semantic_alignment": rel,
                "answer_depth": a_conf,
                "clarity": q_conf,
                "pii_status": "PASSED",
                "duplicate_penalty": penalty_applied,
                "final_weighted_score": score,
            },
        }


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Conversation → FAQ Extractor", layout="wide")
st.title("Conversation → FAQ Extractor")
st.caption("Upload a transcript, see normalization, PII redaction, and FAQ suggestions step by step.")

st.sidebar.header("Configuration")
openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password", help="Used for FAQ extraction.")
model = st.sidebar.selectbox(
    "Model",
    ["gpt-5-nano", "gpt-5-mini", "gpt-4.1-mini", "gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"],
    index=0,
)

uploads = st.sidebar.file_uploader(
    "Upload transcript files (.txt, .csv, .json)", type=["txt", "csv", "json"], accept_multiple_files=True
)
manual_text = st.sidebar.text_area("Or paste raw transcript text")
sample_choice = st.sidebar.selectbox(
    "Sample conversation (optional)",
    ["None"] + list(SAMPLE_TRANSCRIPTS.keys()),
    index=0,
    help="Pick a preset transcript to see the pipeline without uploading files.",
)

raw_lines: List[str] = []
if sample_choice != "None":
    sample_content = SAMPLE_TRANSCRIPTS[sample_choice].encode("utf-8")
    raw_lines.extend(parse_transcript(sample_content, f"{sample_choice}.txt"))

if uploads:
    for f in uploads:
        raw_lines.extend(parse_transcript(f.read(), f.name))
if manual_text.strip():
    raw_lines.extend(
        [
            sanitize_text(line)
            for line in manual_text.splitlines()
            if line.strip()
        ]
    )

if not raw_lines:
    st.info("Upload a transcript file or paste text to begin.")
    st.stop()

# Pipeline steps
normalized_df = normalize_lines(raw_lines)
redacted_df = redact_pii(normalized_df)

cleaned_transcript = "\n".join([f"{row.speaker}: {row.cleaned_text}" for row in redacted_df.itertuples()])

faq_results: List[Dict[str, Any]] = []
faq_error: Optional[str] = None
scored_faqs: List[Dict[str, Any]] = []
raw_llm_response: Optional[str] = None
MAX_TRANSCRIPT_CHARS = 8000
transcript_used = cleaned_transcript
if len(cleaned_transcript) > MAX_TRANSCRIPT_CHARS:
    transcript_used = cleaned_transcript[:MAX_TRANSCRIPT_CHARS]
    st.warning(
        f"Transcript truncated to {MAX_TRANSCRIPT_CHARS} characters for extraction. "
        "Upload fewer/shorter files if you need full coverage."
    )
if st.sidebar.button("Run FAQ Extraction", type="primary"):
    with st.spinner("Extracting FAQs via OpenAI..."):
        try:
            faq_results, raw_llm_response = extract_faq_via_llm(transcript_used, model, openai_api_key)
            if raw_llm_response:
                raw_llm_response = sanitize_text(raw_llm_response)
            if not faq_results:
                faq_error = "No FAQs returned or unable to parse JSON."
            else:
                scorer = FAQScorer()
                for item in faq_results:
                    enriched = {
                        "question": sanitize_text(item.get("question", "")),
                        "answer": sanitize_text(item.get("answer", "")),
                        "confidence": float(item.get("confidence", 0) or 0),
                        # Defaults for scoring; can be extended with real signals
                        "question_confidence": float(item.get("confidence", 0) or 0),
                        "answer_confidence": float(item.get("confidence", 0) or 0),
                        "relevance_score": float(item.get("confidence", 0) or 0),
                        "completeness_score": float(item.get("confidence", 0) or 0),
                        "redundancy_score": 0.0,
                        "pii_removed": True,
                    }
                    decision = scorer.score_faq(enriched)
                    enriched["scorecard"] = decision
                    scored_faqs.append(enriched)
        except Exception as exc:  # pylint: disable=broad-except
            faq_error = str(exc)

# Layout
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Step 1: Raw Transcript")
    st.write("\n".join(raw_lines))

    st.subheader("Step 2: PII Redaction")
    st.dataframe(redacted_df[["speaker", "raw_text", "cleaned_text"]], use_container_width=True, hide_index=True)

with right_col:
    st.subheader("Step 1b: Normalized View")
    st.dataframe(normalized_df, use_container_width=True, hide_index=True)

    st.subheader("Step 3: FAQ Extraction")
    if faq_error:
        st.error(faq_error)
        if raw_llm_response:
            with st.expander("Raw LLM response"):
                st.code(raw_llm_response, language="json")
    elif scored_faqs:
        for item in scored_faqs:
            q = item.get("question", "")
            a = item.get("answer", "")
            conf = float(item.get("confidence", 0) or 0)
            scorecard = item.get("scorecard", {})
            overall = float(scorecard.get("overall_faq_score", 0) or 0)
            decision = scorecard.get("decision", "REJECT")
            color = color_by_confidence(overall)
            with st.expander(f"{decision} · score {overall:.2f}", expanded=False):
                st.markdown(f"**Question:** {q}")
                st.markdown(f"**Answer:** {a}")
                st.markdown(f"**Confidence (LLM):** {conf:.2f}")
                st.markdown(
                    f"**Overall FAQ Score:** <span style='color:{color}; font-weight:bold'>{overall:.2f}</span> ({decision})",
                    unsafe_allow_html=True,
                )
                dbg = scorecard.get("debug_breakdown", {})
                st.markdown("**Debug breakdown**")
                st.json(dbg)
        if raw_llm_response:
            with st.expander("Raw LLM response"):
                st.code(raw_llm_response, language="json")
    else:
        st.info("Run FAQ extraction from the sidebar to see results.")

# Download
if scored_faqs:
    faq_df = pd.DataFrame(scored_faqs)
    csv_buf = io.StringIO()
    faq_df.to_csv(csv_buf, index=False)
    st.download_button("Download FAQ CSV", data=csv_buf.getvalue(), file_name="faqs.csv", mime="text/csv")
