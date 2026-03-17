import sys
from unittest.mock import MagicMock

# Mock the crawlers.data_gateway module
crawlers = MagicMock()
crawlers.data_gateway = MagicMock()
sys.modules['crawlers'] = crawlers
sys.modules['crawlers.data_gateway'] = crawlers.data_gateway
