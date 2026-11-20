
import streamlit as st

from app.services.auth import register_user, authenticate_user


def load_authentication_styles():
    """Apply the StressSense clean, modern light theme."""

    st.markdown(
        """
        <style>
        /* ---------- Global page ---------- */

        .stApp {
            background: #F1F5F9 !important;
            color: #102A43;
        }

        [data-testid="stHeader"] {
            background: transparent;
        }


        [data-testid="stMainBlockContainer"] {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 100%;
        }

        /* ---------- Main content ---------- */

        .auth-wrapper {
            max-width: 440px;
            margin: 2rem auto 0 auto;
            padding: 2.5rem 2.5rem 2rem 2.5rem;
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 20px;
            box-shadow: 0 12px 40px rgba(16, 42, 67, 0.07);
        }

        .brand-mark {
            width: 54px;
            height: 54px;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto 16px auto;
            border-radius: 16px;
            background: #E6F0FA;
            color: #176B87;
            font-size: 27px;
        }

        .brand-title {
            color: #102A43;
            font-size: 32px;
            font-weight: 750;
            letter-spacing: -1.2px;
            text-align: center;
            margin: 0;
        }

        .brand-subtitle {
            color: #627D98;
            font-size: 14px;
            text-align: center;
            margin-top: 7px;
            margin-bottom: 28px;
        }

        .form-heading {
            color: #102A43;
            font-size: 22px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 5px;
        }

        .form-description {
            color: #627D98;
            font-size: 13px;
            text-align: center;
            margin-bottom: 25px;
        }

        .disclaimer {
            max-width: 440px;
            margin: 18px auto 0 auto;
            color: #829AB1;
            font-size: 11px;
            line-height: 1.6;
            text-align: center;
        }

        /* ---------- Input fields ---------- */

        div[data-testid="stTextInput"] label {
            color: #334E68 !important;
            font-size: 13px !important;
            font-weight: 600 !important;
        }

        div[data-testid="stTextInput"] input {
            background: #F8FAFC !important;
            color: #102A43 !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 9px !important;
            min-height: 43px !important;
        }

        div[data-testid="stTextInput"] input:focus {
            border-color: #176B87 !important;
            box-shadow: 0 0 0 1px #176B87 !important;
        }

        /* ---------- Buttons ---------- */

        div.stButton > button,
        div[data-testid="stFormSubmitButton"] > button {
            width: 100%;
            min-height: 44px;
            border-radius: 9px;
            border: 1px solid #102A43;
            background: #102A43;
            color: #FFFFFF;
            font-weight: 650;
            transition: all 0.2s ease;
        }

        div.stButton > button:hover,
        div[data-testid="stFormSubmitButton"] > button:hover {
            background: #176B87;
            border-color: #176B87;
            color: #FFFFFF;
        }

        /* ---------- Links and messages ---------- */

        .switch-text {
            color: #627D98;
            font-size: 13px;
            text-align: center;
            margin-top: 12px;
        }

        [data-testid="stAlert"] {
            border-radius: 9px;
        }

        /* ---------- Responsive layout ---------- */

        @media (max-width: 600px) {
            .auth-wrapper {
                margin: 0.5rem auto;
                padding: 1.5rem;
            }

            .brand-title {
                font-size: 28px;
            }
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def show_brand():
    """Display the StressSense brand identity."""

    st.markdown(
        """
        <div class="brand-mark">⌁</div>
        <h1 class="brand-title">StressSense</h1>
        <p class="brand-subtitle">
            Understand stress through heart rate variability
        </p>
        """,
        unsafe_allow_html=True
    )


def show_login():
    """Display the login form."""

    st.markdown(
        """
        <div class="form-heading">Welcome back</div>
        <div class="form-description">
            Sign in to access your StressSense dashboard.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("login_form"):
        email = st.text_input(
            "Email address",
            placeholder="you@example.com"
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Enter your password"
        )

        submitted = st.form_submit_button(
            "Sign in",
            use_container_width=True
        )

        if submitted:
            if not email.strip() or not password:
                st.error("Please enter both your email and password.")

            else:
                success, message, user = authenticate_user(
                    email.strip(),
                    password
                )

                if success:
                    st.session_state.authenticated = True
                    st.session_state.user = user
                    st.session_state.auth_mode = "Login"

                    st.success("Login successful!")
                    st.rerun()

                else:
                    st.error(message)

    st.markdown(
        '<p class="switch-text">New to StressSense?</p>',
        unsafe_allow_html=True
    )

    if st.button("Create an account", key="go_to_register"):
        st.session_state.auth_mode = "Register"
        st.rerun()


def show_registration():
    """Display the registration form."""

    st.markdown(
        """
        <div class="form-heading">Create your account</div>
        <div class="form-description">
            Join StressSense and explore your HRV patterns.
        </div>
        """,
        unsafe_allow_html=True
    )

    with st.form("registration_form"):
        username = st.text_input(
            "Full name",
            placeholder="Enter your name"
        )

        email = st.text_input(
            "Email address",
            placeholder="you@example.com"
        )

        password = st.text_input(
            "Create password",
            type="password",
            placeholder="At least 8 characters"
        )

        confirm_password = st.text_input(
            "Confirm password",
            type="password",
            placeholder="Enter your password again"
        )

        submitted = st.form_submit_button(
            "Create account",
            use_container_width=True
        )

        if submitted:
            if not all([
                username.strip(),
                email.strip(),
                password,
                confirm_password
            ]):
                st.error("Please fill in all the fields.")

            elif password != confirm_password:
                st.error("Passwords do not match.")

            else:
                success, message, user_id = register_user(
                    username.strip(),
                    email.strip(),
                    password
                )

                if success:
                    st.success("Account created successfully!")
                    st.info("You can now sign in using your credentials.")

                    st.session_state.auth_mode = "Login"
                    st.rerun()

                else:
                    st.error(message)

    st.markdown(
        '<p class="switch-text">Already have an account?</p>',
        unsafe_allow_html=True
    )

    if st.button("Back to sign in", key="go_to_login"):
        st.session_state.auth_mode = "Login"
        st.rerun()


def show_authentication():
    """Render the StressSense authentication screen."""

    load_authentication_styles()

    if "auth_mode" not in st.session_state:
        st.session_state.auth_mode = "Login"

    # Center the authentication card.
    _, center_column, _ = st.columns([1, 1.35, 1])

    with center_column:
        show_brand()

        if st.session_state.auth_mode == "Register":
            show_registration()
        else:
            show_login()

    st.markdown(
        """
        <div class="disclaimer">
            StressSense is an academic research prototype for HRV-based
            stress prediction. It is not a medical diagnostic tool.
        </div>
        """,
        unsafe_allow_html=True
    )
