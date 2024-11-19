from langserve import RemoteRunnable

if __name__ == '__main__':
    # Update the URL to use port 8001 and include /invoke
    client = RemoteRunnable('http://localhost:8001/chainDemo')
    print(client.invoke({
        'language': 'italian',
        'text': 'Hello'
    }))