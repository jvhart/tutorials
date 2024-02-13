# app.py
#
# Use this sample code to handle webhook events in your integration.
#
# 1) Paste this code into a new file (app.py)
#
# 2) Install dependencies
#   pip3 install flask
#   pip3 install stripe
#
# 3) Run the server on http://localhost:4242
#   python3 -m flask run --port=4242

import stripe

from flask import Flask, jsonify, request

# import requests
# requests.post(url='http://127.0.0.1:4242')

# The library needs to be configured with your account's secret key.
# Ensure the key is kept out of any version control system you might be using.
stripe.api_key = "sk_test_51OUWYwHigLPi6Wx67IlfUF2Ovhs3MbI6AvfqLbYJCmc835TGyh59Wpzu20MtLZF7kDQ9nxfdqJMzCeiJLWxQLPUi00M41D0dRC"

# This is your Stripe CLI webhook secret for testing your endpoint locally.
endpoint_secret = 'whsec_bfa79b71657c62ebb7ed4157aacb2fe512b47cc6ec43456432c74a47cc1bf299'

app = Flask(__name__)

@app.route('/webhook', methods=['POST'])
def webhook():
	event = None
	payload = request.data
	sig_header = request.headers['STRIPE_SIGNATURE']

	try:
		event = stripe.Webhook.construct_event(
			payload, sig_header, endpoint_secret
		)
	except ValueError as e:
		# Invalid payload
		raise e
	except stripe.error.SignatureVerificationError as e:
		# Invalid signature
		raise e
	
	# Handle the event
	if event['type'] == 'payment_intent.succeeded':
		payment_intent = event['data']['object']
		# ... handle other event types
	if event['type'] == 'subscription_schedule.canceled':
		subscription_schedule = event['data']['object']
		# ... handle other event types
	if event['type'] == 'invoice.upcoming':
		invoice = event['data']['object']
		# ... handle other event types
	if event['type'] == 'charge.captured':
		charge = event['data']['object']
		# ... handle other event types
	if event['type'] == 'invoice.payment_succeeded':
		invoice = event['data']['object']
		# ... handle other event types
	else:
		print('Unhandled event type {}'.format(event['type']))
	
	return jsonify(success=True)