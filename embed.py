from foundry_local_sdk import Configuration, FoundryLocalManager
from db import init_db, insert_document, get_all_documents

FoundryLocalManager.initialize(Configuration(app_name="rag-assistant"))
manager = FoundryLocalManager.instance

embedding_model = manager.catalog.get_model("qwen3-embedding-0.6b")

if not embedding_model.is_cached:
    print("Downloading embedding model...")
    embedding_model.download(lambda p: print(f"\r%{p:.0f}", end="", flush=True))
    print()

embedding_model.load()

embedding_client = embedding_model.get_embedding_client()

# --- Knowledge base: Albert Camus's philosophy, summarized from the
#     Stanford Encyclopedia of Philosophy entry (paraphrased, not quoted) ---
documents = [
    "Camus's philosophy of the absurd starts from a paradox: humans "
    "cannot stop asking what the meaning of life is, yet the universe "
    "offers no answer. This gap between our need for meaning and the "
    "world's silence is what Camus calls 'the absurd'. He illustrates "
    "it with the image of Sisyphus endlessly pushing a rock up a "
    "mountain, only to watch it roll back down every time.",

    "In The Myth of Sisyphus, Camus argues that recognizing life as "
    "absurd does not justify suicide. Instead, he claims that Sisyphus "
    "can be imagined as happy: by fully accepting his fate and "
    "continuing his task with lucid awareness rather than false hope, "
    "he takes ownership of an existence with no ultimate meaning.",

    "In The Rebel, Camus shifts his focus from suicide to murder, "
    "asking whether killing can ever be justified once we accept the "
    "absurd. He introduces the idea of 'revolt' — a stance where the "
    "individual refuses to accept oppression while also refusing to "
    "claim that any cause justifies unlimited violence.",

    "Camus was strongly critical of Communism and revolutionary "
    "ideology, arguing that movements which believe history guarantees "
    "eventual human happiness end up excusing mass violence as a "
    "necessary cost. He believed revolt should have limits, and warned "
    "that once a rebellion becomes a revolution seeking total control, "
    "it betrays its own original impulse.",

    "Camus rejected being labeled an existentialist and criticized "
    "thinkers like Sartre, Kierkegaard, and Heidegger for what he saw "
    "as an escape from the absurd — turning to religion, history, or "
    "abstract systems to find meaning after all. This disagreement "
    "with Sartre eventually caused a public and personal falling out "
    "between the two during the Cold War.",

    # --- Free will, summarized from the Stanford Encyclopedia of
    #     Philosophy entry on Free Will (paraphrased, not quoted) ---
    "Free will is generally understood as a kind of control over "
    "one's own choices and actions — being the source of what one "
    "does, and often also having the ability to have chosen "
    "otherwise. Philosophers link it closely to moral responsibility: "
    "we tend to think people deserve praise or blame only if their "
    "actions were genuinely up to them.",

    "Compatibilism is the view that free will is compatible with "
    "determinism — the idea that every event, including human "
    "choices, is the inevitable result of prior causes and the laws "
    "of nature. Compatibilists argue that what matters for freedom is "
    "not whether our actions were causally determined, but whether "
    "they flowed from our own reasoning and desires rather than "
    "external coercion.",

    "Libertarianism about free will (not to be confused with the "
    "political philosophy) holds that free will requires that our "
    "choices not be fully determined by prior causes. Libertarians "
    "differ on what more is needed: some appeal to genuine "
    "indeterminism in how reasons cause action, while 'agent-causal' "
    "libertarians argue that the agent herself, not just her mental "
    "states, must be an irreducible source of the choice.",

    "Free will skeptics argue that we may not have free will at all. "
    "Some, like Galen Strawson, contend that being truly responsible "
    "for an action would require being responsible for the character "
    "that produced it, which leads to an impossible infinite regress. "
    "Others point to empirical findings, such as Benjamin Libet's "
    "experiments on brain activity preceding conscious decisions, as "
    "evidence that conscious willing may come later than we assume.",

    # --- Virginia Woolf, verified via web search and paraphrased
    #     (not quoted) from academic and encyclopedia sources ---
    "Virginia Woolf is best known for pioneering the stream of "
    "consciousness technique, most famously in her 1925 novel Mrs "
    "Dalloway. Rather than following external events in order, the "
    "narration drifts through a character's private thoughts, "
    "memories, and sensory impressions as they naturally occur, often "
    "moving between past and present within a single paragraph.",

    "In her 1929 essay A Room of One's Own, Woolf argues that a woman "
    "needs money and a private room in order to write fiction. Her "
    "central claim is that intellectual freedom depends on material "
    "independence — without financial security and physical space of "
    "her own, a woman's creative work is constantly interrupted or "
    "made impossible by economic dependence on others.",

    "Woolf was a central figure in the Bloomsbury Group, a circle of "
    "writers, artists, and intellectuals — including E.M. Forster, "
    "Lytton Strachey, and the economist John Maynard Keynes — who met "
    "informally in London's Bloomsbury district from around 1905 "
    "onward. The group was known for challenging Victorian-era social "
    "conventions, particularly around gender and sexuality.",

    "Born in London in 1882, Virginia Woolf became one of the most "
    "influential modernist writers of the twentieth century, known "
    "for novels such as Mrs Dalloway, To the Lighthouse, and Orlando. "
    "Along with her husband Leonard Woolf, she founded the Hogarth "
    "Press in 1917, which published her own work as well as that of "
    "other emerging writers. She died in 1941.",
]

# Make sure the table exists
init_db()

# Embed all documents in one call
doc_result = embedding_client.generate_embeddings(documents)

# Insert each (content, embedding) pair into SQLite
for doc, item in zip(documents, doc_result.data):
    insert_document(doc, item.embedding)

print(f"Inserted {len(documents)} chunks into rag.db")

# Sanity check: read them back
stored = get_all_documents()
print(f"\nDatabase now contains {len(stored)} rows:")
for row in stored:
    print(f"  id={row['id']}  content={row['content'][:50]!r}  vector_len={len(row['embedding'])}")

embedding_model.unload()