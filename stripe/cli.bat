
stripe login

stripe products create --name="My First Product" --description="Created with the Stripe CLI"

rem product id = prod_PJ9GEuiOn7jfOe

stripe prices create --unit-amount=3000 --currency=usd --product=prod_PJ9GEuiOn7jfOe

stripe listen --forward-to localhost:4242/webhook

stripe trigger payment_intent.succeeded

stripe trigger subscription_schedule.canceled
