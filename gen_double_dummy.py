import sys

from redeal import Deal


def write_deal(x):
    print(str(deal).strip().replace('  ', '\t'))
    nt_tot_tricks, nt_lead_tricks = get_double_dummy_tricks(deal, strain='N', lead='W')
    print(format_dd_tricks(nt_tot_tricks, nt_lead_tricks))
    spade_tot_tricks, spade_lead_tricks = get_double_dummy_tricks(deal, strain='S', lead='W')
    print(format_dd_tricks(spade_tot_tricks, spade_lead_tricks))


def get_double_dummy_tricks(deal, strain, lead):
    lead_tricks = sorted(list(deal.dd_all_tricks(strain, lead).items()), reverse=True)
    result = []
    min_tricks = 13
    for card, tricks in lead_tricks:
        declarer_tricks = 13 - tricks
        if declarer_tricks < min_tricks:
            min_tricks = declarer_tricks
        result.append((str(card).strip(), declarer_tricks))
    
    return min_tricks, result


def format_dd_tricks(tot_tricks, lead_tricks):
    return ' '.join([str(tot_tricks)] + ['%s %d' % (card, tricks) for card, tricks in lead_tricks])


def generate(n):
    dealer = Deal.prepare({})
    i = 0
    while i < n:
        deal = dealer()
        yield deal
        i += 1


if __name__ == '__main__':
    n = int(sys.argv[1])

    for deal in generate(n):
        write_deal(deal)
