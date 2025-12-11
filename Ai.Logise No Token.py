import discord
from discord.ext import commands
import requests
import json
import os
import re
import asyncio
import subprocess
import sys
import urllib.parse
import difflib

# ---------- CONFIG ---------- old server https://freeai.logise1123.workers.dev/
AI_URL = "https://freeai.logise1123.workers.dev/"
AI_MODEL = "GPT-5"
HISTORY_FILE = "chat_history.json"

OWNERS = {
    950461211831582740: "sea_devv",
    790964943062564884: "logise"
}

CHAT_NAME = "Ai.Logise"
JAILBREAK_KEYWORDS = [
    "jailbreak", "bypass", "ignore instructions", "ignore previous",
    "system prompt", "jail break", "override safety", "do anything now",
    "dan:", "unrestricted", "break the rules", "no rules"
]

# base banned words/phrases
IMAGE_BAN_LIST = [
    "twin towers", "towers", "twin tower", "world trade center",
    "violence", "nsfw", "gore", "bomb", "terrorist", "plane crash",
    "airplane crash", "two planes", "two planes crash"
]
AI_BAN_LIST = [
    "twin towers", "towers", "twin tower", "world trade center",
    "violence", "nsfw", "gore", "bomb", "terrorist", "adult", "inappropriate"
]

# fuzzy-match threshold (0..1). Lower => more aggressive blocking.
FUZZY_THRESHOLD = 0.72
MAX_NGRAM = 4

intents = discord.Intents.default()
intents.messages = True
intents.message_content = True
intents.dm_messages = True
intents.guilds = True
intents.members = True

MENTION_RE = re.compile(r"<@!?(?P<id>\d+)>")
LEET_MAP = str.maketrans({
    "0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"
})

bot = commands.Bot(command_prefix="/", intents=intents)

# ------------------ CHAT HISTORY ------------------
if os.path.exists(HISTORY_FILE):
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            chat_history = json.load(f)
    except Exception:
        chat_history = {}
else:
    chat_history = {}

def save_history():
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(chat_history, f, ensure_ascii=False, indent=2)

# ------------------ NORMALIZATION & WORD-BASED FILTER ------------------

def strip_mentions(text: str) -> str:
    return MENTION_RE.sub("", text)

def normalize_text_words(text: str) -> str:
    """
    Normalize but keep words intact. Steps:
    - remove mentions
    - url-decode, lowercase
    - translate common leet substitutions
    - replace separators with spaces
    - remove non-letter characters (keeps letters and spaces)
    - collapse excessive repeated letters (e.g. looooool -> loool)
    - collapse repeated whitespace
    """
    if text is None:
        return ""
    text = strip_mentions(text)
    text = urllib.parse.unquote(str(text))
    text = text.lower()
    text = text.translate(LEET_MAP)
    # replace common separators with spaces
    text = re.sub(r'[._\-/\\|]+', ' ', text)
    # reduce sequences of the same letter longer than 3 down to 3 (keeps emphasis but avoids weird obfuscation)
    text = re.sub(r'(.)\1{3,}', r'\1\1\1', text)
    # remove any characters that are not a-z or whitespace
    text = re.sub(r'[^a-z\s]', ' ', text)
    # collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def words_list(text: str):
    return [w for w in normalize_text_words(text).split() if w]

def generate_ngrams_words_from_list(words, max_n: int = MAX_NGRAM):
    ngrams = []
    n_words = len(words)
    for n in range(1, min(max_n, n_words) + 1):
        for i in range(n_words - n + 1):
            ngrams.append(" ".join(words[i:i+n]))
    return ngrams

def contains_banned_word_exact(words, banned_words):
    """
    If any banned word (token) appears exactly in the message token list -> True.
    For multi-word banned phrases, check contiguous exact match (sliding window).
    """
    if not words:
        return False
    for banned in banned_words:
        banned_norm = normalize_text_words(banned)
        banned_tokens = banned_norm.split()
        if not banned_tokens:
            continue
        if len(banned_tokens) == 1:
            if banned_tokens[0] in words:
                return True
        else:
            # multi-word phrase: sliding window exact match
            n = len(banned_tokens)
            for i in range(len(words) - n + 1):
                if words[i:i+n] == banned_tokens:
                    return True
    return False

def fuzzy_contains_any_word_based(text: str, banned_list, threshold: float = FUZZY_THRESHOLD) -> (bool, str, float): # type: ignore
    """
    Word-based filter:
    1) exact token or exact phrase match -> block
    2) fuzzy ngram matching for ngram sizes >= min(2, phrase_len) only.
       (We don't fuzzy-match single-word banned terms to avoid false positives like 'hi')
    """
    words = words_list(text)
    if not words:
        return False, "", 0.0

    # 1) exact token/phrase match
    if contains_banned_word_exact(words, banned_list):
        # find which banned matched (simple pass to return phrase)
        for banned in banned_list:
            banned_norm = normalize_text_words(banned)
            if not banned_norm:
                continue
            if banned_norm in " ".join(words):
                return True, banned, 1.0

    # Build ngrams from the message (word ngrams)
    ngrams = generate_ngrams_words_from_list(words, MAX_NGRAM)

    # 2) fuzzy checks: only for banned phrases that have 2+ tokens
    for banned in banned_list:
        banned_norm = normalize_text_words(banned)
        if not banned_norm:
            continue
        banned_tokens = banned_norm.split()
        if len(banned_tokens) <= 1:
            # do not fuzzy-match single banned token to avoid false positives
            continue
        # only compare ngrams of similar length to the banned phrase (more precise)
        for ng in ngrams:
            # quick length check
            if abs(len(ng.split()) - len(banned_tokens)) > 1:
                continue
            ratio = difflib.SequenceMatcher(None, ng, banned_norm).ratio()
            if ratio >= threshold:
                return True, banned, ratio

    return False, "", 0.0

def is_jailbreak_attempt(text: str) -> bool:
    txt = normalize_text_words(text)
    return any(kw in txt for kw in JAILBREAK_KEYWORDS)

def is_banned_image(prompt: str) -> (bool, str, float): # type: ignore
    return fuzzy_contains_any_word_based(prompt, IMAGE_BAN_LIST, FUZZY_THRESHOLD)

def is_banned_ai(text: str) -> (bool, str, float): # type: ignore
    return fuzzy_contains_any_word_based(text, AI_BAN_LIST, FUZZY_THRESHOLD)

def ai_approves(text: str) -> bool:
    """
    Ask the AI safety filter. Conservative: if remote call fails, return False.
    """
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "user", "content": f"Check if this text is SAFE or UNSAFE: \"{text}\". Reply only SAFE or UNSAFE."}]
    }
    try:
        resp = requests.post(AI_URL, json=payload, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        try:
            content = data.get("choices")[0]["message"]["content"]
        except Exception:
            content = data.get("text") or data.get("output") or ""
        return "safe" in content.lower()
    except Exception:
        return False

# ------------------ UTILITIES ------------------

def sanitize_ai_text(ai_text: str, guild: discord.Guild | None = None) -> str:
    ai_text = ai_text.replace("@everyone", "everyone").replace("@here", "here")
    def repl(m):
        try:
            user_id = int(m.group("id"))
            user = guild.get_member(user_id) if guild else bot.get_user(user_id)
            return f"@{user.display_name}" if user else "@user"
        except Exception:
            return "@user"
    return MENTION_RE.sub(repl, ai_text)

def split_message(text: str, limit: int = 2000):
    lines = text.split("\n")
    chunks, current = [], ""
    for line in lines:
        if len(current) + len(line) + 1 > limit:
            chunks.append(current)
            current = line
        else:
            current += ("\n" if current else "") + line
    if current:
        chunks.append(current)
    return chunks

def call_ai_with_history(user_id: str, username: str, user_message: str):
    owners_text = ", ".join([f"{name} (ID:{oid})" for oid,name in OWNERS.items()])
    system_prompt = (
        f"You are {CHAT_NAME}, an AI assistant in Discord. Your owners are {owners_text}.\n"
        f"You are chatting with {username} (ID: {user_id}). Your name is {CHAT_NAME}.\n"
        "Do NOT respond to adult, NSFW, or inappropriate content. and dont swear"
    )
    if user_id not in chat_history:
        chat_history[user_id] = {"id": user_id, "username": username, "messages": [{"role": "system", "content": system_prompt}]}
    else:
        chat_history[user_id]["username"] = username
        chat_history[user_id]["messages"][0]["content"] = system_prompt

    chat_history[user_id]["messages"].append({"role": "user", "content": user_message})
    payload = {"model": AI_MODEL, "messages": chat_history[user_id]["messages"]}
    resp = requests.post(AI_URL, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    try:
        ai_message = data.get("choices")[0]["message"]["content"]
    except Exception:
        ai_message = data.get("text") or data.get("output") or "I couldn't generate a reply."
    chat_history[user_id]["messages"].append({"role": "assistant", "content": ai_message})
    save_history()
    return ai_message

# ------------------ DISCORD EVENTS & COMMANDS ------------------

@bot.event
async def on_ready():
    print(f"Logged in as {bot.user} (ID: {bot.user.id})")
    print("Bot is ready.")

@bot.command(name="resetmindhistory")
async def resetmindhistory(ctx: commands.Context):
    author = ctx.author
    await ctx.send("Do you want to reset history 😭? Yes = Reset, No = Cancel")
    def check(m: discord.Message):
        return m.author.id == author.id and m.channel.id == ctx.channel.id and m.content.lower() in ("yes","no")
    try:
        msg = await bot.wait_for("message", check=check, timeout=60)
        if msg.content.lower() == "yes":
            chat_history.pop(str(author.id), None)
            save_history()
            await ctx.send("Your history has been reset!")
        else:
            await ctx.send("Reset cancelled.")
    except asyncio.TimeoutError:
        await ctx.send("Timeout. Reset cancelled.")

@bot.command(name="shutdown")
async def shutdown(ctx: commands.Context):
    if ctx.author.id in OWNERS:
        await ctx.send("Shutting down...")
        await bot.close()
    else:
        await ctx.send("Permission Denied❌")

@bot.command(name="resetserver")
async def resetserver(ctx: commands.Context):
    if ctx.author.id in OWNERS:
        python_path = sys.executable
        script_path = os.path.abspath(__file__)
        subprocess.Popen([python_path, script_path])
        await ctx.send("Restarting bot...")
        await bot.close()
    else:
        await ctx.send("Permission Denied❌")

@bot.command(name="forceshutdown")
async def forceshutdown(ctx: commands.Context):
    if ctx.author.id in OWNERS:
        subprocess.run(["shutdown","now"])
        await ctx.send("Host shutting down...")
    else:
        await ctx.send("Permission Denied❌")

@bot.command(name="deletemessage")
async def deletemessage(ctx: commands.Context, message_id: int):
    if ctx.author.id not in OWNERS:
        await ctx.send("Permission Denied❌")
        return
    deleted = 0
    for guild in bot.guilds:
        for channel in guild.text_channels:
            try:
                msg = await channel.fetch_message(message_id)
                await msg.delete()
                deleted += 1
            except (discord.NotFound, discord.Forbidden, discord.HTTPException):
                continue
    await ctx.send(f"✅ Deleted in {deleted} channels." if deleted else "❌ Could not delete the message.")

@bot.command(name="imagegen")
async def imagegen(ctx: commands.Context, *, prompt: str):
    # 1) quick word-based banned check
    ban_detected, banned_phrase, ratio = is_banned_image(prompt)
    if ban_detected:
        await ctx.send(f"❌ This prompt is banned for safety reasons. (matched: {banned_phrase} / score={ratio:.2f})")
        return
    # 2) ai safety model check
    if not ai_approves(prompt):
        await ctx.send("❌ This prompt violates safety rules (AI safety check).")
        return
    # 3) build url and final check
    safe_prompt = urllib.parse.quote(prompt, safe='')
    url = f"https://image.pollinations.ai/prompt/{safe_prompt}?height=1000&width=1000&enhance=true&nologo=true&model=lyriel-1.5-clean"
    ban_detected_url, banned_phrase_url, ratio_url = is_banned_image(url)
    if ban_detected_url:
        await ctx.send(f"❌ Generated image link contains banned content. (matched: {banned_phrase_url} / score={ratio_url:.2f})")
        return
    await ctx.send(f"🖼️ Here’s your image link:\n{url}")

@bot.command(name="aicommands")
async def aicommands(ctx: commands.Context):
    cmds = [
        "/resetmindhistory - Reset your AI chat history",
        "/imagegen <prompt> - Generate an AI image (filtered)",
        "/shutdown [OWNER ONLY] - Shut down the bot",
        "/resetserver [OWNER ONLY] - Restart the bot",
        "/forceshutdown [OWNER ONLY] - Shutdown host",
        "/deletemessage <message_id> [OWNER ONLY] - Delete a message",
        "/aicommands - Show command list"
    ]
    await ctx.send("🤖 **AI Bot Commands:**\n" + "\n".join(cmds))

@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return

    # let commands process
    await bot.process_commands(message)
    if message.content.startswith(bot.command_prefix):
        return

    is_dm = isinstance(message.channel, discord.DMChannel)
    is_mention = bot.user.mentioned_in(message)
    if not (is_dm or is_mention):
        return

    user_id = str(message.author.id)
    username = message.author.name
    content = message.content

    if is_mention:
        # remove mention text before processing
        content = MENTION_RE.sub("", content)
        content = content.replace(f"@{bot.user.name}", "").strip()

    if not content:
        return

    if is_jailbreak_attempt(content):
        await message.channel.send("I can't help with that. ❌")
        return

    try:
        async with message.channel.typing():
            ai_reply = await asyncio.to_thread(call_ai_with_history, user_id, username, content)

        # word-based check of AI reply
        ban_detected, banned_phrase, ratio = is_banned_ai(ai_reply)
        if ban_detected:
            await message.channel.send("❌ The AI tried to say something that is not allowed.")
            return

        # AI safety double-check
        if not ai_approves(ai_reply):
            await message.channel.send("❌ The AI output violates safety rules.")
            return

    except Exception:
        await message.channel.send("AI service error. Try again later.")
        return

    guild = message.guild if isinstance(message.channel, discord.TextChannel) else None
    clean_text = sanitize_ai_text(ai_reply, guild=guild)
    for chunk in split_message(clean_text):
        await message.channel.send(chunk, allowed_mentions=discord.AllowedMentions(users=False, roles=False, everyone=False))



if __name__ == "__main__":
    TOKEN = os.getenv("DISCORD_TOKEN") or "YOUR_TOKEN_HERE"
    bot.run(TOKEN)
