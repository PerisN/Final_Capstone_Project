
import hashlib
import hmac

EMAIL = "wanjiru456@gmail.com"
PASSWORD = "niggasaintshit"

def verify_login(email, password):
    email_matches = hmac.compare_digest(
        email.strip().lower(),
        EMAIL.strip().lower()
    )

    password_matches = hmac.compare_digest(
        password,
        PASSWORD
    )

    return email_matches and password_matches