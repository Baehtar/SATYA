from bot.bot import create_bot


def main():
    print("🚀 Starting Satya...")

    application = create_bot()

    print("🤖 Satya is running!")
    print("Press Ctrl+C to stop.")

    application.run_polling(bootstrap_retries=-1)


if __name__ == "__main__":
    main()