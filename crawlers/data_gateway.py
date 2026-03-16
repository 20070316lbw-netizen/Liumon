class Gateway:
    def get_stock_data(self, ticker, start, end):
        return "Date,Open,High,Low,Close,Volume\n2023-01-01,100,105,95,102,1000\n2023-01-02,102,108,100,105,1200\n"
gateway = Gateway()
