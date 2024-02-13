
# https://stripe.com/docs/development

# whsec_bfa79b71657c62ebb7ed4157aacb2fe512b47cc6ec43456432c74a47cc1bf299

import stripe

# STRIPE_PUBLISHABLE_KEY='pk_test_51OUWYwHigLPi6Wx6DdO0dMicxOKFOljfYTtBU7tMWjCfTsmtqaPRqwd1nq4ucyHGFJ9mSWmjS6LU4NetvpZlvon600uARVSSgP'

stripe.api_key = "sk_test_51OUWYwHigLPi6Wx67IlfUF2Ovhs3MbI6AvfqLbYJCmc835TGyh59Wpzu20MtLZF7kDQ9nxfdqJMzCeiJLWxQLPUi00M41D0dRC"

starter_subscription = stripe.Product.create(
  name="Starter Subscription",
  description="$12/Month subscription",
)

starter_subscription_price = stripe.Price.create(
  unit_amount=1200,
  currency="usd",
  recurring={"interval": "month"},
  product=starter_subscription['id'],
)

# Save these identifiers
print(f"Success! Here is your starter subscription product id: {starter_subscription.id}")
# Success! Here is your starter subscription product id: prod_PJ9eIzjiUq87vs
print(f"Success! Here is your starter subscription price id: {starter_subscription_price.id}")
# Success! Here is your starter subscription price id: price_1OUXLNHigLPi6Wx6cOqJpfI9