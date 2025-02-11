ollama pull nomic-embed-text - memory first step
install postgreSQL and add it to the path

IN TERMINAL
psql -U postgres

CREATE USER qargo WITH PASSWORD '5787' SUPERUSER;
CREATE DATABASE memory_agent;
GRANT ALL PRIVILEGES ON SCHEMA public TO qargo;
GRANT ALL PRIVILEGES ON DATABASE memory_agent TO qargo;
\c memory_agent
CREATE TABLE conversations (
id SERIAL PRIMARY KEY,
timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
prompt TEXT NOT NULL,
response TEXT NOT NULL
);

INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'what is my name?', 'Y
our name is Dima. Known online as Qargo.');

INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'What is 3355 / 15?',
'223.666667');

INSERT INTO conversations (timestamp, prompt, response) VALUES (CURRENT_TIMESTAMP, 'What do i like?', 'You like Anime, cats, tech and your dreams');

SELECT * FROM conversations;

ollama local model:
# zephyr-ollama/Modelfile
FROM ./zephyr-ollama
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
TEMPLATE """{% for message in messages %}{{message['role']}}: {{message['content']}}{% endfor %}"""

# Build model package
ollama create zephyr -f ./zephyr-ollama/Modelfile