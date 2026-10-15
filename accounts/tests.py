from django.test import TestCase
from django.urls import reverse
from django.contrib.auth.models import User


class RegisterFlowTests(TestCase):
	def test_new_user_redirects_to_login_after_registration(self):
		response = self.client.post(
			reverse("accounts:register"),
			{
				"first_name": "John",
				"last_name": "Doe",
				"username": "johndoe",
				"email": "john@example.com",
				"password1": "StrongPass123",
				"password2": "StrongPass123",
			},
			follow=False,
		)

		self.assertEqual(response.status_code, 302)
		self.assertRedirects(response, reverse("accounts:login"))
		self.assertNotIn("_auth_user_id", self.client.session)

	def test_new_user_can_sign_in_and_open_dashboard(self):
		self.client.post(
			reverse("accounts:register"),
			{
				"first_name": "John",
				"last_name": "Doe",
				"username": "newuser",
				"email": "newuser@example.com",
				"password1": "StrongPass123",
				"password2": "StrongPass123",
			},
		)

		response = self.client.post(
			reverse("accounts:login"),
			{"username": "newuser", "password": "StrongPass123"},
		)

		self.assertRedirects(response, reverse("dashboard:home"))
		self.assertEqual(
			self.client.session["_auth_user_id"],
			str(User.objects.get(username="newuser").pk),
		)
		self.assertEqual(
			self.client.get(reverse("dashboard:home")).status_code,
			200,
		)

	def test_existing_user_can_sign_in_and_open_dashboard(self):
		user = User.objects.create_user(
			username="previoususer",
			password="StrongPass123",
		)

		response = self.client.post(
			reverse("accounts:login"),
			{"username": "previoususer", "password": "StrongPass123"},
		)

		self.assertRedirects(response, reverse("dashboard:home"))
		self.assertEqual(
			self.client.session["_auth_user_id"],
			str(user.pk),
		)
		self.assertEqual(
			self.client.get(reverse("dashboard:home")).status_code,
			200,
		)

	def test_user_can_log_out_and_switch_accounts(self):
		previous_user = User.objects.create_user(
			username="previoususer",
			password="StrongPass123",
		)
		new_user = User.objects.create_user(
			username="newuser",
			password="StrongPass123",
		)

		self.client.force_login(previous_user)
		logout_response = self.client.post(reverse("accounts:logout"))
		self.assertRedirects(
			logout_response,
			reverse("dashboard:landing"),
			fetch_redirect_response=False,
		)
		self.assertNotIn("_auth_user_id", self.client.session)

		login_response = self.client.post(
			reverse("accounts:login"),
			{"username": "newuser", "password": "StrongPass123"},
		)
		self.assertRedirects(login_response, reverse("dashboard:home"))
		self.assertEqual(
			self.client.session["_auth_user_id"],
			str(new_user.pk),
		)
