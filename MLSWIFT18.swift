import NaturalLanguage

let tagger = NLTagger(tagSchemes: [.sentimentScore])
let options: NLTagger.Options = [.omitPunctuation]
let string = "This is a great product!"
tagger.string = string
tagger.enumerateTags(in: string.startIndex..<string.endIndex, unit: .word, scheme: .sentimentScore, options: options) { tag, range in
    if let sentimentScore = tag?.rawValue as? NSSentimentScore {
        print(sentimentScore)
    }
}
