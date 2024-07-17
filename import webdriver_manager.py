import webdriver_manager
from selenium import webdriver

# Install the ChromeDriver using webdriver_manager
webdriver_manager.ChromeDriver().install()

# Create a new ChromeDriver instance
driver = webdriver.Chrome()

# Use the driver to navigate to a website
driver.get("https://www.example.com")

# Close the driver
driver.quit()