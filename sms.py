print("SMS Sent Successfully :)")

import uuid
import base64
import boto3
import requests
import requests

def send_sms(message,mob_no):
	url = "https://www.fast2sms.com/dev/bulkV2"
	payload = "sender_id=FSTSMS&message="+ message +"&language=english&route=p&numbers="+ mob_no +""
	headers = {
	 'authorization': "tGjIAhpz1L57S3aZ6leyCixgQWfHvK20VRrnDFsuUb9qPN8Jkd8qfThBuliVJM2Yx3QO9PcyEGesNd7I",
	 'Content-Type': "application/x-www-form-urlencoded",
	 'Cache-Control': "no-cache",
	 }
	response = requests.request("POST", url, data=payload, headers=headers)
	print(response.text)

# send_sms("Crying","8605883174")
